import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Set page configuration
st.set_page_config(page_title="Color Perception Data Analysis", layout="wide")

# Title
st.title("Color Perception Data Analysis")

st.write("""
Upload the CSV data exported from the Nature's Palette study to analyze participant responses and visualize the results.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read data
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")

        # Check if required columns exist
        required_columns = ['selected_color_space', 'object']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            st.error(f"The following required columns are missing from the data: {', '.join(missing_columns)}")
            st.stop()

        if st.checkbox("Show Raw Data"):
            st.subheader("Raw Data")
            st.dataframe(data)

        # Begin Analysis
        st.header("Analysis 1: Frequency of Selections for Each Color Format")
        color_counts = data['selected_color_space'].value_counts()
        st.subheader("Total Selections by Color Format")

        # Explanation
        st.write("""
        This chart shows the number of times each color format (CMYK, Pantone, RGB) was selected by participants across all tasks. It helps identify which color formats are generally preferred.
        """)

        fig1, ax1 = plt.subplots()
        sns.barplot(x=color_counts.index, y=color_counts.values, ax=ax1, palette='Set2')
        ax1.set_xlabel('Color Format')
        ax1.set_ylabel('Number of Selections')
        ax1.set_title('Frequency of Selections for Each Color Format')
        st.pyplot(fig1)

        # Analysis 2: Preferred Color Formats Across Participants
        st.header("Analysis 2: Preferred Color Formats Across Participants")
        color_percentages = color_counts / color_counts.sum() * 100
        st.subheader("Percentage of Selections by Color Format")

        # Explanation
        st.write("""
        This table and pie chart display the percentage of total selections that each color format received. This indicates the relative popularity of each color format among all participants.
        """)

        st.dataframe(color_percentages.to_frame(name='Percentage (%)'))

        fig2, ax2 = plt.subplots()
        ax2.pie(color_counts.values, labels=color_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2', len(color_counts)))
        ax2.axis('equal')
        ax2.set_title('Percentage of Selections by Color Format')
        st.pyplot(fig2)

        # Analysis 3: Preferred Color Formats by Object
        st.header("Analysis 3: Preferred Color Formats by Object")
        object_color = data.groupby(['object', 'selected_color_space']).size().unstack(fill_value=0)
        st.subheader("Selections per Object and Color Format")

        # Explanation
        st.write("""
        This table and heatmap show how often each color format was selected for each object. It helps determine if certain objects have a preferred color format, indicating the importance of color accuracy for specific objects.
        """)

        st.dataframe(object_color)

        fig3, ax3 = plt.subplots(figsize=(10,8))
        sns.heatmap(object_color, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_xlabel('Color Format')
        ax3.set_ylabel('Object')
        ax3.set_title('Heatmap of Color Format Selection by Object')
        st.pyplot(fig3)

        # Function to calculate Cramér's V
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

        # Analysis 4: Statistical Significance Testing
        st.header("Analysis 4: Statistical Significance Testing")
        st.subheader("Chi-Squared Test for Overall Color Format Preferences")

        # Explanation
        st.write("""
        The chi-squared test determines whether there is a statistically significant difference in the frequency of selections among color formats. A significant result suggests that participants have a preference for certain color formats.
        """)

        if len(color_counts) > 1 and color_counts.sum() > 0:
            chi2_stat, p_value, dof, expected = chi2_contingency([color_counts.values])
            cramer_v_value = cramers_v(np.array([color_counts.values]))
            st.write(f"Chi-squared Statistic: {chi2_stat:.2f}")
            st.write(f"P-value: {p_value:.4f}")
            st.write(f"Cramér's V (Effect Size): {cramer_v_value:.4f}")
            if p_value < 0.05:
                st.success("The differences in color format selections are statistically significant (p < 0.05).")
            else:
                st.info("The differences in color format selections are not statistically significant (p ≥ 0.05).")
        else:
            st.write("Not enough categories or data for chi-squared test.")

        # Chi-Squared Test for each object
        st.subheader("Chi-Squared Test for Color Preferences by Object")

        # Explanation
        st.write("""
        This test examines whether the selection frequencies of color formats differ significantly for each object. It helps identify objects where color format preferences are especially pronounced.
        """)

        for obj in object_color.index:
            obs = object_color.loc[[obj]].values
            total = obs.sum()
            if obs.shape[1] > 1 and total > 0 and (obs > 0).all():
                try:
                    chi2_stat, p_value, dof, expected = chi2_contingency(obs)
                    cramer_v_value = cramers_v(obs)
                    st.write(f"**Object:** {obj}")
                    st.write(f"Chi-squared Statistic: {chi2_stat:.2f}")
                    st.write(f"P-value: {p_value:.4f}")
                    st.write(f"Cramér's V (Effect Size): {cramer_v_value:.4f}")
                    if p_value < 0.05:
                        st.success(f"For {obj}, the differences in color format selections are statistically significant (p < 0.05).")
                    else:
                        st.info(f"For {obj}, the differences in color format selections are not statistically significant (p ≥ 0.05).")
                except Exception as e:
                    st.write(f"**Object:** {obj}")
                    st.write(f"Chi-squared test not valid: {e}")
            else:
                st.write(f"**Object:** {obj}")
                st.write("Not enough data or categories for chi-squared test, or zero counts present.")
            st.write("---")

        # Additional Analysis: Participant Consistency Over Repeats
        st.header("Additional Analysis: Participant Consistency Over Repeats")

        # Explanation
        st.write("""
        This analysis checks whether participants consistently selected the same color format for the same object across different repeats. A higher consistency rate suggests strong preferences or perceptions regarding color formats for specific objects.
        """)

        if 'user_id' in data.columns and 'repeat' in data.columns:
            user_objects = data.groupby(['user_id', 'object'])['selected_color_space'].agg(list).reset_index()
            user_objects['consistent'] = user_objects['selected_color_space'].apply(lambda x: len(set(x)) == 1)
            st.subheader("Participant Consistency Data")
            st.dataframe(user_objects)

            consistency_rate = user_objects['consistent'].mean() * 100
            st.write(f"Overall Consistency Rate: {consistency_rate:.2f}%")

            # Plotting consistency
            st.subheader("Consistency Rate Visualization")
            fig4, ax4 = plt.subplots()
            labels = ['Consistent', 'Inconsistent']
            sizes = [user_objects['consistent'].sum(), len(user_objects) - user_objects['consistent'].sum()]
            ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
            ax4.axis('equal')
            st.pyplot(fig4)
        else:
            st.write("Participant consistency analysis requires 'user_id' and 'repeat' columns in the data.")
    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
        st.stop()
else:
    st.info("Please upload a CSV file to proceed with the analysis.")