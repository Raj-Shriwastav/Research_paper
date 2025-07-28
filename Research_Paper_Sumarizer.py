import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from io import BytesIO
import re

deepgram_api = "" #<- API here

# Set page config
st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="wide")

# Streamlit app header
st.title("📄 PDF Summarizer with Chat and Feedback")

# Function to get API key
def get_api_key():
    api_key = "AIzaSyBdKj1y4K-SPLWpMEA7x1Hboxsg4R47M5w"
    return api_key

# Gemini model setup
def text_model(api_key):
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.9)

# Fine-tuning prompt
def fine_tuning():
    return """
    You are an AI summarizer with excellent knowledge in summarizing content. 
    You will be given a large amount of content. Your task is to summarize the content provided to you in the best possible way by identifying 
    
    important points and highlighting key terms.

    Additionally in the response include the existing diagrams and graphs present in the content as well as same diagrams and graphs.

    Out of the above article I provided you please construct out the following things as an article for me first A title of the article in between the range of 10 to 15 words A description of the article in 45 to 55 words And a detailed reconstruction of the article focusing upon the conclusion of the entire invention, Discovery or break through and the discussion upon what possibilities it brings into the real world in range of 490 to 510 words keeping the primary focus upon the concept and miarly touching the person who did this the institution where it was done and the approach that was taken focus centrally on the advancement that happened.

    Additionally when you create the output make sure it is in this format:
    title : 
    description:
    details description:
    """

# Function to extract text from PDF
def get_pdf_content(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to summarize text
def summarize(pdf_content, api_key):
    prompt = fine_tuning() + "\n\nHere's the content to summarize:\n" + pdf_content
    message = HumanMessage(content=prompt)
    llm = text_model(api_key)
    response = llm.invoke([message])
    return response.content

# Function to process bold text within *__*
def process_bold_text(line):
    """Convert text enclosed in *__* to bold."""
    bold_pattern = re.compile(r'\**__(.*?)__\**')
    processed_line = bold_pattern.sub(r'<b>\1</b>', line)
    return processed_line

def create_enhanced_pdf(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.steelblue,
        alignment=TA_CENTER,
        spaceAfter=20,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.steelblue,
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'Subheading',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.darkgrey,
        spaceBefore=6,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['BodyText'],
        fontSize=12,
        textColor=colors.black,
        spaceBefore=6,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )

    # Process the text
    sections = text.split('\n')
    is_first_paragraph = True
    for i, section in enumerate(sections):
        section = section.strip()
        if i == 0:  # Main title
            title_text = section.lstrip('#').strip()  # Remove leading '#'
            story.append(Paragraph(title_text, title_style))
        elif section.startswith('##') or section.startswith('###'):  # Subheading or sub-subheading
            if not is_first_paragraph:
                story.append(Spacer(1, 12))  # Extra space before new section
            if section.startswith('## '):  # Subheading
                subheading_text = section.lstrip('#').strip()  # Remove leading '## '
                story.append(Paragraph(subheading_text, heading_style))
            elif section.startswith('### '):  # Sub-subheading
                subheading_text = section.lstrip('#').strip()  # Remove leading '### '
                story.append(Paragraph(subheading_text, subheading_style))
            is_first_paragraph = True
        elif section:  # Body text
            # Process bold text
            processed_line = re.sub(r'\**(.*?)\**', r'<b>\1</b>', section)
            if is_first_paragraph:
                story.append(Paragraph(processed_line, body_style))
                is_first_paragraph = False
            else:
                story.append(Spacer(1, 12))  # Extra space between paragraphs
                story.append(Paragraph(processed_line, body_style))
        else:  # Empty line
            if not is_first_paragraph:
                story.append(Spacer(1, 12))  # Extra space for empty lines
                is_first_paragraph = True

    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    api_key = get_api_key()
    
    if not api_key:
        st.warning("Please set your Google API Key as an environment variable to proceed.")
        return

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'summary_stage' not in st.session_state:
        st.session_state.summary_stage = "initial"

    # Main summarization section
    st.header("PDF Summarizer")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Summarize") or st.session_state.summary_stage == "modify":
            with st.spinner("📚 Summarizing... Please wait."):
                # Extract text from PDF
                pdf_content = get_pdf_content(uploaded_file)
                
                if st.session_state.summary_stage == "initial":
                    # Initial summarization
                    st.session_state.summary = summarize(pdf_content, api_key)
                elif st.session_state.summary_stage == "modify":
                    # Modify summary based on user feedback
                    modification_prompt = st.session_state.modification_prompt
                    new_prompt = f"Please modify the following summary according to this instruction: {modification_prompt}\n\nOriginal summary:\n{st.session_state.summary}"
                    st.session_state.summary = summarize(new_prompt, api_key)

                # Display summary
                st.subheader("Summary:")
                st.write(st.session_state.summary)

                # Ask for user feedback
                st.session_state.summary_stage = "feedback"

    if st.session_state.summary_stage == "feedback":
        user_satisfied = st.radio("Are you satisfied with this summary?", ("Yes", "No"))
        
        if user_satisfied == "Yes":
            # Provide download button
            pdf_buffer = create_enhanced_pdf(st.session_state.summary)
            st.download_button(
                label="Download Summary PDF",
                data=pdf_buffer,
                file_name="summary.pdf",
                mime="application/pdf"
            )
            
            # Reset summary stage
            st.session_state.summary_stage = "initial"
            
            # Initialize chat history with the summary
            st.session_state.chat_history = [AIMessage(content=st.session_state.summary)]
        
        elif user_satisfied == "No":
            st.session_state.modification_prompt = st.text_area("How should the summary be modified?")
            if st.button("Modify Summary"):
                st.session_state.summary_stage = "modify"
                st.rerun()

    # Download button for the last modified content
    if st.session_state.summary:
        st.subheader("Download Latest Summary")
        pdf_buffer = create_enhanced_pdf(st.session_state.summary)
        st.download_button(
            label="Download Latest Summary PDF",
            data=pdf_buffer,
            file_name="latest_summary.pdf",
            mime="application/pdf"
        )

    # Chat interface
    st.header("Chat about the Summary")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # Chat input
    user_input = st.chat_input("Type your message here:")
    if user_input:
        with st.spinner("🤔 Thinking..."):
            # Add user message to chat history
            st.session_state.chat_history.append(HumanMessage(content=user_input))

            # Get AI response
            llm = text_model(api_key)
            conversation = ConversationChain(
                llm=llm,
                memory=ConversationBufferMemory(return_messages=True)
            )
            
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    conversation.predict(input=message.content)
                elif isinstance(message, AIMessage):
                    conversation.memory.chat_memory.add_ai_message(message.content)

            response = conversation.predict(input=user_input)

            # Add AI response to chat history
            st.session_state.chat_history.append(AIMessage(content=response))

        # Refresh the page to show the updated chat
        st.rerun()

if __name__ == "__main__":
    main()
