import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
def getLLamaresponse(input_text,no_words,blog_style):

    ### LLama2 model
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    
    ## Prompt Template
    template = """
    Write a professional blog for {blog_style} about the topic "{input_text}".
    Explain clearly, use simple language and make it engaging.
    The blog should be within {no_words} words.
    """

    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response = llm.invoke(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    response = response.content
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("🤖 AI Blog Generator")
st.write("Enter a topic and generate a professional blog using LLM")

input_text = st.text_input("📝 Enter Blog Topic")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientist','Common People'),index=0)
    
submit = st.button("🚀 Generate Blog")

## Final response
if submit:
    with st.spinner("Generating blog..."):
        response = getLLamaresponse(input_text,no_words,blog_style)
        st.write(response)
