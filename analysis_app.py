import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Color Perception Data Analysis", layout="wide")
st.title("Color Perception Data Analysis")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")

    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(data)

    st.header("Analysis 1: Frequency of Selections for Each Color Format")
    color_counts = data['selected_color_space'].value_counts()
    st.subheader("Total Selections by Color Format")
    fig1, ax1 = plt.subplots()
    sns.barplot(x=color_counts.index, y=color_counts.values, ax=ax1, palette='Set2')
    ax1.set_xlabel('Color Format')
    ax1.set_ylabel('Number of Selections')
    ax1.set_title('Frequency of Selections for Each Color Format')
    st.pyplot(fig1)

    st.header("Analysis 2: Preferred Color Formats Across Participants")
    color_percentages = color_counts / color_counts.sum() * 100
    st.subheader("Percentage of Selections by Color Format")
    st.write(color_percentages.to_frame(name='Percentage (%)'))

    fig2, ax2 = plt.subplots()
    ax2.pie(color_counts.values, labels=color_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2', len(color_counts)))
    ax2.axis('equal')
    ax2.set_title('Percentage of Selections by Color Format')
    st.pyplot(fig2)

    st.header("Analysis 3: Preferred Color Formats by Object")
    object_color = data.groupby(['object', 'selected_color_space']).size().unstack(fill_value=0)
    st.subheader("Selections per Object and Color Format")
    st.dataframe(object_color)

    fig3, ax3 = plt.subplots(figsize=(10,8))
    sns.heatmap(object_color, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel('Color Format')
    ax3.set_ylabel('Object')
    ax3.set_title('Heatmap of Color Format Selection by Object')
    st.pyplot(fig3)

    def cramers_v(confusion_matrix):
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2 / n
        r,k = confusion_matrix.shape
        with np.errstate(divide='ignore', invalid='ignore'):
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
            if np.isnan(v):
                return 0.0
            else:
                return v

    st.header("Analysis 4: Statistical Significance Testing")
    st.subheader("Chi-Squared Test for Overall Color Format Preferences")

    if len(color_counts) > 1 and color_counts.sum() > 0:
        chi2_stat, p_value, dof, expected = chi2_contingency([color_counts.values])
        cramer_v_value = cramers_v(np.array([color_counts.values]))
        st.write(f"Chi-squared Statistic: {chi2_stat:.2f}")
        st.write(f"P-value: {p_value:.4f}")
        st.write(f"Cramér's V: {cramer_v_value:.4f}")
        if p_value < 0.05:
            st.success("The differences in color format selections are statistically significant.")
        else:
            st.info("The differences in color format selections are not statistically significant.")
    else:
        st.write("Not enough categories or data for chi-squared test.")

    st.subheader("Chi-Squared Test for Color Preferences by Object")
    for obj in object_color.index:
        obs = object_color.loc[obj].values
        total = obs.sum()
        if len(obs) > 1 and total > 0 and (obs > 0).all():
            try:
                chi2_stat, p_value, dof, expected = chi2_contingency([obs])
                cramer_v_value = cramers_v(np.array([obs]))
                st.write(f"**Object:** {obj}")
                st.write(f"Chi-squared Statistic: {chi2_stat:.2f}")
                st.write(f"P-value: {p_value:.4f}")
                st.write(f"Cramér's V: {cramer_v_value:.4f}")
                if p_value < 0.05:
                    st.success(f"For {obj}, the differences in color format selections are statistically significant.")
                else:
                    st.info(f"For {obj}, the differences in color format selections are not statistically significant.")
            except ValueError as e:
                st.write(f"**Object:** {obj}")
                st.write(f"Chi-squared test not valid: {e}")
        else:
            st.write(f"**Object:** {obj}")
            st.write("Not enough data or categories for chi-squared test, or zero counts present.")
        st.write("---")

    if 'user_id' in data.columns:
        st.header("Additional Analysis: Participant Consistency Over Repeats")
        user_objects = data.groupby(['user_id', 'object'])['selected_color_space'].agg(list).reset_index()
        user_objects['consistent'] = user_objects['selected_color_space'].apply(lambda x: len(set(x)) == 1)
        st.subheader("Participant Consistency Data")
        st.dataframe(user_objects)

        consistency_rate = user_objects['consistent'].mean() * 100
        st.write(f"Overall Consistency Rate: {consistency_rate:.2f}%")
    else:
        st.write("Participant consistency analysis requires 'user_id' in the data.")

else:
    st.info("Please upload a CSV file to proceed with the analysis.")