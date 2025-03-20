import streamlit as st
import re
from groq import Groq
from datetime import datetime
import pandas as pd
from textblob import TextBlob  # For sentiment analysis
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator  # For translation

# Initialize Groq client
client = Groq(api_key="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")  # Replace with your actual API key

# Simulated Database for Candidate Data (In-memory storage)
if "candidate_database" not in st.session_state:
    st.session_state.candidate_database = []

# Model class for LLM integration using Groq
class GroqModel:
    def __init__(self):
        self.client = client
    
    def generate_response(self, messages):
        """
        Generate a response from the LLM based on the given messages using Groq API.
        """
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
        return response.choices[0].message.content

# Hiring Assistant class
class HiringAssistant:
    def __init__(self):
        self.model = GroqModel()
        self.exit_keywords = ["exit", "quit", "bye", "goodbye", "end"]
        self.required_info = ["name", "email", "phone", "experience", "position", "location", "tech_stack"]
    
    def get_greeting(self):
        greeting = """
        Hello and welcome to TalentScout! My name is Emily, and I'm your hiring assistant here at TalentScout. 
        It's a pleasure to connect with you today.  

        Before we dive in, I'd like to confirm that you're interested in exploring exciting technology job opportunities 
        with our clients. My goal is to help you find the perfect role that aligns with your skills, experience, 
        and career aspirations.  

        To get started, could you please tell me your full name? This will help me personalize our conversation 
        and ensure I address you correctly.  

        Once I have your name, I'll ask a few questions about your background, expertise, and what you're looking 
        for in your next career move. Let's work together to find the best opportunities for you!
        """
        return greeting
    
    def get_farewell(self):
        prompt = """
        You are a hiring assistant for TalentScout.
        The candidate is ending the conversation.
        Provide a warm, professional farewell message.
        Thank them for their time and mention that the TalentScout team will be in touch if there's a suitable match.
        """
        return self.model.generate_response([{"role": "user", "content": prompt}])
    
    def is_exit_keyword(self, text):
        return any(keyword in text.lower() for keyword in self.exit_keywords)
    
    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    def detect_language(self, text):
        try:
            if len(text.strip().split()) >= 3:
                detected_lang = detect(text)
                return detected_lang
            else:
                return "en"
        except Exception as e:
            print(f"Error detecting language: {e}")
            return "en"
    
    def translate_text(self, text, src_lang, dest_lang="en"):
        try:
            translated = GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
            return translated
        except Exception as e:
            print(f"Error translating text: {e}")
            return text
    
    def process_input(self, user_input, current_state, candidate_info, asked_questions):
        # Detect language and translate input to English
        src_lang = self.detect_language(user_input)
        user_input_en = self.translate_text(user_input, src_lang, "en") if src_lang != "en" else user_input
        
        # Analyze sentiment
        sentiment_score = self.analyze_sentiment(user_input_en)
        
        # Generate a context-aware response
        messages = [
            {"role": "system", "content": f"""
            You are a hiring assistant for TalentScout. The current state of the conversation is: {current_state}. 
            The candidate has provided the following information so far: {candidate_info}. 
            Respond appropriately based on the user input: {user_input_en}.
            
            The sentiment analysis of the candidate's input is: {"positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"}.
            Adjust your tone accordingly to be empathetic and supportive if the sentiment is negative, or enthusiastic if the sentiment is positive.
            """},
            {"role": "user", "content": user_input_en}
        ]
        response_en = self.model.generate_response(messages)
        
        # Translate response back to the candidate's language
        response = self.translate_text(response_en, "en", src_lang) if src_lang != "en" else response_en
        
        # Extract information from the user input
        extracted_info = self._extract_info(user_input_en, candidate_info)
        candidate_info.update(extracted_info)
        
        # Determine the next question based on missing information
        next_question = self._get_next_question(candidate_info)
        
        if next_question:
            return {
                "message": next_question,
                "new_state": "collect_info",
                "candidate_info": candidate_info
            }
        else:
            # All information collected, generate technical questions
            return self.generate_tech_questions(candidate_info["tech_stack"], candidate_info, response)
    
    def _extract_info(self, text, candidate_info):
        extracted_info = {}
        
        # Extract name
        if "name" not in candidate_info:
            name = self._extract_name(text)
            if name:
                extracted_info["name"] = name
        
        # Extract email
        if "email" not in candidate_info:
            email = self._extract_email(text)
            if email:
                extracted_info["email"] = email
        
        # Extract phone
        if "phone" not in candidate_info:
            phone = self._extract_phone(text)
            if phone:
                extracted_info["phone"] = phone
        
        # Extract experience
        if "experience" not in candidate_info:
            experience = self._extract_experience(text)
            if experience is not None:
                extracted_info["experience"] = experience
        
        # Extract position
        if "position" not in candidate_info:
            position = text.strip() if text.strip() else None
            if position:
                extracted_info["position"] = position
        
        # Extract location
        if "location" not in candidate_info:
            location = text.strip() if text.strip() else None
            if location:
                extracted_info["location"] = location
        
        # Extract tech stack
        if "tech_stack" not in candidate_info:
            tech_stack = text.strip() if text.strip() else None
            if tech_stack:
                extracted_info["tech_stack"] = tech_stack
        
        return extracted_info
    
    def _get_next_question(self, candidate_info):
        missing_info = [info for info in self.required_info if info not in candidate_info]
        if missing_info:
            next_info = missing_info[0]
            prompt = f"""
            You are a hiring assistant for TalentScout.
            The candidate has provided the following information so far: {candidate_info}.
            Politely ask the candidate for their {next_info.replace('_', ' ')}.
            """
            return self.model.generate_response([{"role": "user", "content": prompt}])
        else:
            return None
    
    def generate_tech_questions(self, tech_stack, candidate_info, response):
        messages = [
            {"role": "system", "content": f"Generate 3-5 technical questions based on the candidate's tech stack: {tech_stack}. Make the questions relevant, varied in difficulty, and focused on practical application."},
            {"role": "user", "content": "Generate technical questions."}
        ]
        questions = self.model.generate_response(messages)
        return {
            "message": f"Here are some technical questions based on your skills:\n\n{questions}\n\nPlease answer these questions to the best of your ability.",
            "new_state": "tech_questions",
            "candidate_info": candidate_info,
            "asked_questions": questions.split("\n")
        }
    
    def process_tech_answers(self, user_input, candidate_info, asked_questions, response):
        if not hasattr(candidate_info, "tech_answers"):
            candidate_info["tech_answers"] = [user_input]
        else:
            candidate_info["tech_answers"].append(user_input)
        
        # Save candidate data to the simulated database
        candidate_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.candidate_database.append(candidate_info)
        
        return {
            "message": "Thank you for answering the questions! We'll review your responses and get back to you soon.",
            "new_state": "wrap_up",
            "candidate_info": candidate_info
        }
    
    def end_conversation(self, user_input, candidate_info, response):
        return {
            "message": "Thank you for your time! We'll be in touch if there's a suitable match.",
            "new_state": "end",
            "candidate_info": candidate_info
        }
    
    # Helper methods for extracting information
    def _extract_name(self, text):
        words = text.strip().split()
        if len(words) >= 1:
            return " ".join(words)
        return None
    
    def _extract_email(self, text):
        pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        match = re.search(pattern, text)
        if match:
            return match.group()
        return None
    
    def _extract_phone(self, text):
        pattern = r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{10})"
        match = re.search(pattern, text)
        if match:
            return match.group()
        return None
    
    def _extract_experience(self, text):
        pattern = r"(\d+)"
        match = re.search(pattern, text)
        if match:
            return int(match.group())
        return None
# Main Streamlit application
def main():
    st.set_page_config(page_title="TalentScout Hiring Assistant", page_icon="🤖", layout="wide")
    
    st.title("TalentScout Hiring Assistant")
    st.markdown("#### Tech Recruitment Initial Screening")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.candidate_info = {
            "name": None,
            "email": None,
            "phone": None,
            "experience": None,
            "position": None,
            "location": None,
            "tech_stack": None
        }
        st.session_state.current_state = "greeting"
        st.session_state.asked_questions = []
    
    # Initialize the assistant
    assistant = HiringAssistant()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initial greeting
    if not st.session_state.messages:
        greeting = assistant.get_greeting()
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        with st.chat_message("assistant"):
            st.markdown(greeting)
    
    # Get user input
    user_input = st.chat_input("Type your response here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Check for exit keywords
        if assistant.is_exit_keyword(user_input):
            farewell = assistant.get_farewell()
            st.session_state.messages.append({"role": "assistant", "content": farewell})
            with st.chat_message("assistant"):
                st.markdown(farewell)
            return
        
        # Process user input based on current state
        response = assistant.process_input(user_input, st.session_state.current_state, st.session_state.candidate_info, st.session_state.asked_questions)
        
        # Update session state
        st.session_state.current_state = response["new_state"]
        st.session_state.candidate_info = response["candidate_info"]
        if "asked_questions" in response:
            st.session_state.asked_questions = response["asked_questions"]
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["message"]})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response["message"])
        
        # Auto-scroll to bottom
        st.rerun()

    # Display candidate database (for admin purposes)
    if st.sidebar.checkbox("View Candidate Database (Admin)"):
        st.sidebar.write("### Candidate Database")
        if st.session_state.candidate_database:
            df = pd.DataFrame(st.session_state.candidate_database)
            st.sidebar.dataframe(df)
        else:
            st.sidebar.write("No candidate data available yet.")

if __name__ == "__main__":
    main()
