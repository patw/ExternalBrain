# Basic flask stuff for building http APIs and rendering html templates
from flask import Flask, render_template, redirect, url_for, request, session, jsonify

# Bootstrap integration with flask so we can make pretty pages
from flask_bootstrap import Bootstrap

# Flask forms integrations which save insane amounts of time
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, SelectField, TextAreaField, IntegerField, FloatField, DateTimeField
from wtforms.validators import DataRequired

# Basic python stuff
import os
import json
import functools
import requests
import time
import datetime

# Mongo stuff
import pymongo
from bson import ObjectId

# Some nice formatting for code
import misaka

# Nice way to load environment variables for deployments
from dotenv import load_dotenv
load_dotenv()

# Create the Flask app object
app = Flask(__name__)

# Session key
app.config['SECRET_KEY'] = os.environ["SECRET_KEY"]

# API Key
API_KEY = os.environ["API_KEY"]

# User Auth
users_string = os.environ["USERS"]
users = json.loads(users_string)

# Some handy defaults that control the app
DEFAULT_SYSTEM_MESSAGE = "You are a friendly chatbot. You help the user answer questions, solve problems and make plans.  You think deeply about the question and provide a detailed, accurate response."
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SCORE_CUT = 0.92
DEFAULT_LIMIT = 3
DEFAULT_CANDIDATES = 100
DEFAULT_FACTS_PER_PAGE = 25

# Load the llm model config
with open("model.json", 'r',  encoding='utf-8') as file:
    model = json.load(file)

# Load the embedder config
with open("embedder.json", 'r',  encoding='utf-8') as file:
    embedder = json.load(file)

# Connect to mongo using environment variables
client = pymongo.MongoClient(os.environ["MONGO_CON"])
db = client[os.environ["MONGO_DB"]]

# Make it pretty because I can't :(
Bootstrap(app)

# A form for asking your external brain questions
class QuestionForm(FlaskForm):
    question = StringField('Question 💬', validators=[DataRequired()])
    submit = SubmitField('Submit')
    rag = SelectField('Generation Type', choices=[("augmented", "Augmented (Facts)"), ("freeform", "Freeform")])
    temperature = FloatField('LLM Temperature', default=DEFAULT_TEMPERATURE, validators=[DataRequired()])
    candidates = IntegerField('Vector Candidates', default=DEFAULT_CANDIDATES, validators=[DataRequired()])
    limit = IntegerField('Chunks', default=DEFAULT_LIMIT, validators=[DataRequired()])
    score_cut = FloatField('Score Cut Off', default=DEFAULT_SCORE_CUT, validators=[DataRequired()])
    
# A form for testings semantic search of the chunks
class VectorSearchForm(FlaskForm):
    question = StringField('Question 💬', validators=[DataRequired()])
    k = IntegerField("K Value", validators=[DataRequired()])
    score_cut = FloatField("Score Cut Off", validators=[DataRequired()])
    submit = SubmitField('Search')
  
# A form for pasting in your text for summarization
class PasteForm(FlaskForm):
    context = StringField('Context (article name, url, todo, idea)', validators=[DataRequired()])
    paste_data = TextAreaField('Paste in Text ✂️', validators=[DataRequired()])
    submit = SubmitField('Summarize')

# A form for reviewing and saving the summarized facts
class FactSearchForm(FlaskForm):
    query = StringField('Search Facts', validators=[DataRequired()])
    submit = SubmitField('Search')

# A form to edit the facts
class FactEditForm(FlaskForm):
    context = StringField('Context (article name, url, todo, idea)', validators=[DataRequired()])
    fact = StringField('Fact', validators=[DataRequired()])
    save = SubmitField('Save')

# A form for reviewing and saving the summarized facts
class SaveFactsForm(FlaskForm):
    context = StringField('Context (article name, url, todo, idea)', validators=[DataRequired()])
    fact_data = TextAreaField('Summarized Facts (one per line)', validators=[DataRequired()])
    submit = SubmitField('Save Facts')

# A form for configuring the chunk regen process
class ChunksForm(FlaskForm):
    method = SelectField('Chunking Method', choices=[("limit", "Fixed Fact Limit"), ("context", "Context Grouped (TODO)"), ("similar", "Vector Similarity (TODO)")]) 
    fact_limit = IntegerField('Number of facts per chunk', validators=[DataRequired()])
    submit = SubmitField('Generate Chunks')

# Amazing, I hate writing this stuff
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Function to call the text embedder
def embed(text):
    response = requests.get(embedder["embedding_endpoint"], params={"text":text, "instruction": "Represent this text for retrieval:" }, headers={"accept": "application/json"})
    vector_embedding = response.json()
    return vector_embedding

# Function to call the configured model to get a completion
def llm(user_prompt, system_message, temperature=DEFAULT_TEMPERATURE, tokens=-1):

     # Build the prompt
    prompt = model["prompt_format"].replace("{system}", system_message)
    prompt = prompt.replace("{prompt}", user_prompt)

    api_data = {
        "prompt": prompt,
        "n_predict": tokens,
        "temperature": temperature,
        "stop": model["stop_tokens"],
        "seed": -1,
        "tokens_cached": 0
    }

    # Attempt to do a completion but retry and back off if the model is not ready
    retries = 3
    backoff_factor = 1
    while retries > 0:
        try:
            response = requests.post(model["llama_endpoint"], headers={"Content-Type": "application/json"}, json=api_data)
            json_output = response.json()
            output = json_output['content']
            break
        except:
            time.sleep(backoff_factor)
            backoff_factor *= 2
            retries -= 1
            output = "My AI model is not responding try again in a moment 🔥🐳"
            continue
    
    # Send the raw completion output content
    return output

# Chat with model with or without augmentation
def chat(prompt, system_message, augmented=True, temperature=DEFAULT_TEMPERATURE, candidates=DEFAULT_CANDIDATES, limit=DEFAULT_LIMIT, score_cut=DEFAULT_SCORE_CUT):
    # If we're doing RAG, vector search, assemble chunks and query with them
    fact_chunks = []
    chunk_string = ""
    if augmented:
        chunks = search_chunks(prompt, candidates, limit, score_cut)
        for chunk in chunks:
            fact_chunks.append(chunk["fact_chunk"])
            chunk_string += chunk["fact_chunk"]
        # The most important guardrail for any LLM app:  Don't answer the question if there's no chunks!
        if chunk_string != "":
            llm_prompt = F"Facts:\n{chunk_string}\nAnswer this question using only the relevant facts above: {prompt}"
        else:
            return {"chunks": [], "completion": "I couldn't find any stored facts to answer that question. Try another question."}
    # If we're not, just query the model directly, without augmentation
    else:
        llm_prompt = prompt

    completion = llm(llm_prompt, system_message, temperature)
    return {"chunks": fact_chunks, "completion": completion}

# Store a fact array in mongo
def store_facts(facts, user, context):
    col = db["facts"]
    dt = datetime.datetime.now()
    for fact in facts:
        col.insert_one({"user": user, "date": dt, "context": context, "fact": fact })

# Get chunks based on semantic search (FIX THIS!!)
# Just returns all chunks right now...
def search_chunks(prompt, candidates, limit, score_cut):
    # Get the embedding for the prompt first
    vector = embed(prompt)

    # Build the Atlas vector search aggregation
    vector_search_agg = [
        {
            "$vectorSearch": { 
                "index": "default",
                "path": "chunk_embedding",
                "queryVector": vector,
                "numCandidates": candidates, 
                "limit": limit
            }
        },
        {
            "$project": {
                "fact_chunk": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$match": {
                "score": { "$gte": score_cut }
            }
        }
    ]

    # Connect to chunks, run query, return results
    col = db["chunks"]
    chunk_records = col.aggregate(vector_search_agg)
    # Return this as a list instead of a cursor
    return chunk_records

# Get all facts
def get_facts(skip,limit):
    col = db["facts"]
    fact_records = col.find().skip(skip).limit(limit)
    return fact_records

# Get chunks based on semantic search (FIX THIS!!)
# Just returns all chunks right now...
def search_facts(query):
    
    # Build the Atlas vector search aggregation
    search_agg = [
        {
            "$search": {
                "text": {
                    "path": ["context", "fact"],
                    "query": query
                }
            }
        },
        {
            "$project": {
                "_id": 1,
                "user": 1,
                "date": 1,
                "context": 1,
                "fact": 1,
                "score": {"$meta": "searchScore"}
            }
        },
        {
            "$limit": DEFAULT_FACTS_PER_PAGE
        }
    ]

    # Connect to chunks, run query, return results
    col = db["facts"]
    fact_records = col.aggregate(search_agg)
    # Return this as a list instead of a cursor
    return fact_records

# Return the count of facts in the system
def count_facts():
    col = db["facts"]
    return  col.count_documents({})

# Clean up a list of facts
def clean_facts(facts):            
    # Parse and save the facts here!
    facts = facts.replace("- ", "") # remove bullet points
    facts = facts.replace("* ", "") # also bullet points
    facts = facts.replace("\r", "") # No carraige return
    facts = facts.replace("\t", "") # No tabs

    # Split the facts by line
    fact_list = facts.split("\n")

    # Remove any empty facts from the list
    facts_clean = list(filter(None, fact_list))

    return(facts_clean)

# Generate chunk collection using facts, fixed fact method
def chunk_by_limit(chunk_limit):
    facts_col = db["facts"]
    fact_records = facts_col.find()

    # Drop old chunks to make new chunks!
    chunks_col = db["chunks"]
    chunks_col.delete_many({})  # Goodbye chunks!

    # We detect the changes in user/context so we indicate who said what about what and when
    fact_user = ""
    fact_context = ""
    chunk_string = ""
    fact_count = 0
    for fact in fact_records:
        if fact["user"] != fact_user or fact["context"] != fact_context:
            fact_user = fact["user"]
            fact_context = fact["context"]
            fact_date = fact["date"].strftime('%Y-%m-%d')
            chunk_string += F"{fact_user} on {fact_date} said this about {fact_context}:\n"
        fact_data = fact["fact"]
        chunk_string += F"- {fact_data}\n"
        fact_count += 1
        # We have the maximum number of facts now, lets send it to chunks collection
        if fact_count == chunk_limit:
            chunks_col.insert_one({"fact_chunk": chunk_string, "chunk_embedding": embed(chunk_string)})
            # reset it all!
            fact_count = 0 
            chunk_string = ""
            fact_user = ""
            fact_context = ""
    # Clean up final facts
    if chunk_string != "":
        chunks_col.insert_one({"fact_chunk": chunk_string, "chunk_embedding": embed(chunk_string)})

# Generate chunk collection using facts, fixed fact limit but chunks can only contain a single context
def chunk_by_context(chunk_limit):
    # TODO - fix this
    return chunk_by_limit(chunk_by_limit)

# Generate chunk collection using facts, fixed fact limit but chunks will always contain similar facts
def chunk_by_similarity(chunk_limit):
    # TODO - fix this
    return chunk_by_limit(chunk_by_limit)


# Define a decorator to check if the user is authenticated
# No idea how this works...  Magic.
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if users != None:
            if session.get("user") is None:
                return redirect(url_for('login'))
        return view(**kwargs)        
    return wrapped_view

# The default question view
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():

    # Question form for the external brain
    form = QuestionForm()

    # If user is prompting send it
    if form.validate_on_submit():

         # Get the form variables
        form_result = request.form.to_dict(flat=True)
        q = form_result["question"]

        if form_result["rag"] == "augmented":
            llm_result = chat(q, DEFAULT_SYSTEM_MESSAGE, True)
        else:
            llm_result = chat(q, DEFAULT_SYSTEM_MESSAGE, False)

        # Format with Misaka
        formatted_result = misaka.html(llm_result["completion"])
        return render_template('index.html', llm_result=formatted_result, form=form)
    
    # Spit out the template
    return render_template('index.html', llm_result="", form=form)

# The paste in data, get facts, store facts
@app.route('/paste', methods=['GET', 'POST'])
@login_required
def pastetext():
    # Question form for the external brain
    form = PasteForm()

    # Send the text to the llm or the facts to mongo
    if form.is_submitted():
        # Get the form variables
        form_result = request.form.to_dict(flat=True)
        
        if form_result["submit"] == "Save Facts":
            # Parse and save the facts here!
            facts = clean_facts(form_result["fact_data"])
            store_facts(facts, session["user"], form_result["context"])
            return redirect(url_for('index'))
        else:
            paste_data = form_result["paste_data"]
            context =  form_result["context"]
            llm_prompt = F"Provide a detailed fact based summary the following {context}. Output a single level of bullet points and ONLY bullet points not numbers: {paste_data}"
            llm_result = llm(llm_prompt, DEFAULT_SYSTEM_MESSAGE).lstrip().rstrip()

             # Change the form to the fact review form to save facts
            form = SaveFactsForm(fact_data=llm_result, context=form_result["context"])
            return render_template('factreview.html', form=form)
    
    # Spit out the template
    return render_template('pastedata.html', form=form)

# The paste in data, get facts, store facts
@app.route('/manual', methods=['GET', 'POST'])
@login_required
def manual():
    form = SaveFactsForm()

    if form.is_submitted():
        # Get the form variables
        form_result = request.form.to_dict(flat=True)
        
        # Parse and save the facts here!
        facts = clean_facts(form_result["fact_data"])
        store_facts(facts, session["user"], form_result["context"])
        return redirect(url_for('index'))
        
    return render_template('factreview.html', form=form)

# Regenerate the chunks!
@app.route('/facts', methods=['GET', 'POST'])
@login_required
def facts():
    form = FactSearchForm()
    fact_count = count_facts()

    # For paginating the display
    page = request.args.get('page')
    if not page:
        page = 0
    else:
        page = int(page)

    # Make sure the page doesn't go too low or too high
    if page < 0:
        page = 0
    if page >= fact_count / DEFAULT_FACTS_PER_PAGE:
        page = int(fact_count / DEFAULT_FACTS_PER_PAGE)

    if form.is_submitted():
        form_result = request.form.to_dict(flat=True)
        facts = search_facts(form_result["query"])
    else:
        skip = page * DEFAULT_FACTS_PER_PAGE
        limit = DEFAULT_FACTS_PER_PAGE
        facts = get_facts(skip, limit)
    return render_template('facts.html', page=page, form=form, facts=facts, facts_count=fact_count)

# Regenerate the chunks!
@app.route('/chunks', methods=['GET', 'POST'])
@login_required
def regenchunks():
    # Question form for the external brain
    form = ChunksForm(fact_limit=DEFAULT_LIMIT)
    fact_count = count_facts()

    # Regen the chunks based on settings
    if form.is_submitted():
        # Get the form variables
        form_result = request.form.to_dict(flat=True)
        fact_limit = int(form_result["fact_limit"])
        method = form_result["method"]
        if method == "limit":
            chunk_by_limit(fact_limit)
        elif method == "context":
            chunk_by_context(fact_limit)  # FIX THIS!!
        elif method == "similar":
            chunk_by_similarity(fact_limit)  # FIX THISS!!
        return redirect(url_for('index'))

    # Spit out the template
    return render_template('chunks.html', fact_count=fact_count, form=form)

# Search chunks
@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    chunks = []
    form = VectorSearchForm(k=DEFAULT_CANDIDATES, score_cut=DEFAULT_SCORE_CUT)
    # Regen the chunks based on settings
    if form.is_submitted():
        form_result = request.form.to_dict(flat=True)
        # Search chunks using vector search
        prompt = form_result["question"]
        candidates = int(form_result["k"])
        score_cut = float(form_result["score_cut"])
        chunks = search_chunks(prompt, candidates, 10, score_cut)

    return render_template('search.html', chunks=chunks, form=form)


# This fact is wrong and will be corrected harshly
@app.route('/edit/<id>', methods=['GET', 'POST'])
@login_required
def fact_edit(id):
    
    # Pull up that fact
    facts_col = db["facts"]
    fact_record = facts_col.find_one({'_id': ObjectId(id)})

    # Fact edit form
    form = FactEditForm()

    if form.is_submitted():
        form_result = request.form.to_dict(flat=True)
        form_result.pop('csrf_token')
        form_result.pop('save')
        facts_col.update_one({'_id': ObjectId(id)}, {'$set': form_result})
        return redirect(url_for('index'))
    else:        
        # Populate the edit form.
        form.context.data = fact_record["context"]
        form.fact.data = fact_record["fact"]
        return render_template('edit.html', form=form)

# This fact is bad, and it should feel bad
@app.route('/delete/<id>')
@login_required
def fact_delete(id):
    facts_col = db["facts"]
    facts_col.delete_one({'_id': ObjectId(id)})
    return redirect(url_for('facts'))


# API route for chatting with bot
@app.route('/api/chat', methods=['GET'])
def api_query():
    # Get the API key from the URL and bail out if it's wrong
    api_key = request.args.get('api_key')
    if api_key != API_KEY:
        return jsonify({"error": "Invalid API Key"})
    
    # Get the prompt, validate it's something
    prompt = request.args.get('prompt')
    if not prompt:
        return jsonify({"error": "you must provide a prompt"})
    
    # Optional system message tunable
    system_message = request.args.get('system_message')
    if not system_message:
        system_message = DEFAULT_SYSTEM_MESSAGE
    
    # Augmented or raw model?
    augmented = request.args.get('augmented')
    if not augmented:
        augmented = True
    else:
        if augmented == "False" or "false":
            augmented = False
        else:
            augmented = True

    # Spicy or boring?
    temperature = request.args.get('temperature')
    if not temperature:
        temperature = DEFAULT_TEMPERATURE

    # Working hard or hardly working?
    candidates = request.args.get('candidates')
    if not candidates:
        candidates = DEFAULT_CANDIDATES

    # Lots of facts or just a few?
    limit = request.args.get('limit')
    if not limit:
        limit = DEFAULT_LIMIT

    # Highly discerning or sloppy?
    score_cut = request.args.get('score_cut')
    if not score_cut:
        score_cut = DEFAULT_SCORE_CUT

    # Ask for a good result
    response = chat(prompt, system_message, augmented, float(temperature), int(candidates), int(limit), float(score_cut))
    return jsonify(response)

# Login/logout routes that rely on the user being stored in session
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.username.data in users:
            if form.password.data == users[form.username.data]:
                session["user"] = form.username.data
                return redirect(url_for('index'))
    return render_template('login.html', form=form)

# We finally have a link for this now!
@app.route('/logout')
def logout():
    session["user"] = None
    return redirect(url_for('login'))
