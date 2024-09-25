import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Color Perception Data Analysis", layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Color Perception Data Analysis")

st.write("""
Upload the CSV data exported from the Nature's Palette study to analyze participant responses and visualize the results.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")

    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(data, use_container_width=True)

    st.header("Analysis 1: Frequency of Selections for Each Color Format")
    color_counts = data['selected_color_space'].value_counts()
    st.subheader("Total Selections by Color Format")
    fig1, ax1 = plt.subplots()
    ax1.bar(color_counts.index, color_counts.values, color='skyblue')
    ax1.set_xlabel('Color Format')
    ax1.set_ylabel('Number of Selections')
    st.pyplot(fig1)

    st.header("Analysis 2: Preferred Color Formats Across Participants")
    color_percentages = color_counts / color_counts.sum() * 100
    st.subheader("Percentage of Selections by Color Format")
    st.write(color_percentages)

    fig2, ax2 = plt.subplots()
    ax2.pie(color_counts, labels=color_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)

    st.header("Analysis 3: Preferred Color Formats by Object")
    object_color = data.groupby(['object', 'selected_color_space']).size().unstack(fill_value=0)
    st.subheader("Selections per Object and Color Format")
    st.dataframe(object_color, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(10,8))
    sns.heatmap(object_color, annot=True, fmt='d', cmap='Blues', ax=ax3)
    plt.xticks(rotation=45)
    plt.title("Heatmap of Color Format Selection by Object")
    st.pyplot(fig3)

    st.header("Analysis 4: Statistical Significance Testing")

    st.subheader("Chi-Squared Test for Overall Color Format Preferences")
    chi2_stat, p_value, dof, expected = chi2_contingency([color_counts.values])
    st.write(f"Chi-squared Statistic: {chi2_stat:.2f}")
    st.write(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        st.success("The differences in color format selections are statistically significant.")
    else:
        st.info("The differences in color format selections are not statistically significant.")

    st.subheader("Chi-Squared Test for Color Preferences by Object")
    for obj in object_color.index:
        obs = object_color.loc[obj].values.reshape(1, -1)
        chi2_stat, p_value, dof, expected = chi2_contingency(obs)
        st.write(f"**Object:** {obj}")
        st.write(f"Chi-squared Statistic: {chi2_stat:.2f}")
        st.write(f"P-value: {p_value:.4f}")
        if p_value < 0.05:
            st.success(f"For {obj}, the differences in color format selections are statistically significant.")
        else:
            st.info(f"For {obj}, the differences in color format selections are not statistically significant.")
        st.write("---")

    if 'user_id' in data.columns:
        st.header("Additional Analysis: Participant Consistency Over Repeats")
        user_objects = data.groupby(['user_id', 'object'])['selected_color_space'].agg(list).reset_index()
        user_objects['consistent'] = user_objects['selected_color_space'].apply(lambda x: len(set(x)) == 1)
        st.subheader("Participant Consistency Data")
        st.dataframe(user_objects, use_container_width=True)

        consistency_rate = user_objects['consistent'].mean() * 100
        st.write(f"Overall Consistency Rate: {consistency_rate:.2f}%")
    else:
        st.write("Participant consistency analysis requires 'user_id' in the data.")

else:
    st.info("Please upload a CSV file to proceed with the analysis.")