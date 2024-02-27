# Basic flask stuff for building http APIs and rendering html templates
from flask import Flask, render_template, redirect, url_for, request, session, jsonify

# Bootstrap integration with flask so we can make pretty pages
from flask_bootstrap import Bootstrap

# Flask forms integrations which save insane amounts of time
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

# Basic python stuff
import os
import requests

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

# URL Enpoint for the Brain
BRAIN_URL = os.environ["BRAIN_URL"]

# Get some identity info for the bot
BOT_NAME = os.environ["BOT_NAME"]
BOT_IDENTITY = os.environ["BOT_IDENTITY"]
BOT_SKILLS = os.environ["BOT_SKILLS"]

# Make it pretty because I can't :(
Bootstrap(app)

# A form for asking your external brain questions
class QuestionForm(FlaskForm):
    question = StringField('Question 💬', validators=[DataRequired()])
    submit = SubmitField('Submit')

# The default question view
@app.route('/', methods=['GET', 'POST'])
def index():

    # Question form for the external brain
    form = QuestionForm()
    formatted_result = ""

    # If user is prompting send it
    if form.validate_on_submit():

         # Get the form variables
        form_result = request.form.to_dict(flat=True)
        q = form_result["question"]

        params = {
            "api_key": API_KEY,
            "prompt": q,
            "system_message": BOT_IDENTITY
        }
        api_response = requests.get(BRAIN_URL, params=params)
        print(api_response.text)
        chat_data = api_response.json()

        llm_result = chat_data["completion"]

        # Format with Misaka
        formatted_result = misaka.html(llm_result)
    
    # Spit out the template
    return render_template('index.html', llm_result=formatted_result, form=form, bot_name=BOT_NAME, bot_skills=BOT_SKILLS)