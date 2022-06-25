import streamlit as st
import numpy as np
import pickle
import re
import random
import socket
from urllib.parse import urlparse

model = pickle.load(open('p1.pkl', 'rb'))

st.title("Domainify")
st.subheader("Phishing Domain Detector Engine")
with st.form("form1", clear_on_submit=False):
    text_input = st.text_input("Enter a URL")

    if st.form_submit_button("Go"):
        domain = urlparse(text_input).netloc

        text = "AaEeIiOoUu"
        count = [i for i in str(domain) if i in text]
        vowels = len(count)

        length = len(domain)

        address = socket.gethostbyname(domain)
        ip = None
        if address != None:
            ip = 1
        else:
            ip = 0

        lis= [0,1]
        server = random.choice(lis)

        sign = re.findall("[._/?=@&! ,+*#$%]", domain)
        sign_count = len(sign)

        features = [vowels,length,ip,server,sign_count]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)

        output = prediction[0]

        if output == 0:
            st.markdown('The Domin is Legitimate')
        else:
            st.markdown('The Domin is Malicious')