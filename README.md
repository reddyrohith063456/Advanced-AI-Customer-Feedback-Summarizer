# AI-Powered Customer Feedback Summarizer

## Project Overview
This project is an AI-powered web application designed to help businesses analyze and summarize customer feedback efficiently. The app leverages advanced natural language processing techniques using OpenAI’s GPT API combined with sentiment analysis and data visualization to provide actionable insights from raw customer reviews and feedback data.

Built with Streamlit for a smooth and interactive user interface, the application supports uploading CSV files containing customer feedback, processes and cleans the data, performs sentiment classification, generates AI-powered summaries, and presents detailed visualizations. This tool empowers product managers, analysts, and business stakeholders to quickly understand customer sentiments, identify common themes, and make data-driven decisions.

## Demo


https://github.com/user-attachments/assets/22189072-b667-43a1-b280-ce86884be5dd



---

## Key Features
- **Upload customer feedback data** in CSV format with ease.
- **AI-powered summarization** of textual feedback using OpenAI GPT models.
- **Sentiment analysis** classifying feedback as positive, negative, or neutral.
- **Visualization dashboards** showing sentiment distribution, keyword trends, and product-wise analysis.
- **Interactive filtering** options to explore feedback by product and sentiment.
- **Export options** for summarized data and reports.
- **Handles structured and unstructured feedback** across multiple product lines.

---

## Technologies and Tools Used
- **Python 3.x** – Core programming language.
- **Streamlit** – For building the interactive web app interface.
- **OpenAI GPT API** – For natural language processing, summarization, and sentiment classification.
- **Pandas & scikit-learn** – For data manipulation and preprocessing.
- **TextBlob** – For additional sentiment scoring.
- **Matplotlib & Seaborn** – For plotting and data visualization.
- **python-dotenv** – For managing environment variables securely.
- **Git** – Version control.

---

## Getting Started

### Prerequisites
- Python 3.7 or above installed.
- An OpenAI API key. You can obtain one from [OpenAI](https://platform.openai.com/account/api-keys).
- Basic familiarity with command-line interfaces.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/feedback-summarizer.git
   cd feedback-summarizer
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows use: venv\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the root project directory.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

---

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Upload your CSV file** containing customer feedback using the app’s interface.

3. **Interact with the dashboard**:
   - View sentiment summaries.
   - Explore visualizations of sentiment trends and product feedback.
   - Use filters to narrow down insights by product or sentiment type.

4. **Download summarized reports** for presentations or further analysis.

---

## Project Structure

```
feedback-summarizer/
│
├── app.py                 # Main Streamlit app script
├── utils.py               # Utility functions for data processing and analysis
├── prompts.py             # GPT prompt templates and prompt engineering
├── requirements.txt       # List of Python dependencies
├── README.md              # Project documentation
└── .env                   # Environment variables (not committed to repo)
```

---

## Configuration

- **OpenAI API Key**: Required to access GPT services. Must be set in `.env`.
- **File format**: Input feedback data should be in CSV format with at least columns for feedback text and product name.
- **Customizing prompts**: Modify `prompts.py` to tailor AI behavior for different types of summarization or sentiment analysis.

---

## Contributing

Contributions are welcome! Whether it’s bug fixes, new features, or documentation improvements:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request.

Please follow code style conventions and provide clear commit messages.

---

## Troubleshooting & FAQs

- **Q:** What if the app fails to connect to OpenAI API?  
  **A:** Ensure your API key is correct in `.env` and your internet connection is active.

- **Q:** Can I upload large files?  
  **A:** For very large datasets, consider pre-processing to reduce size or split files.

- **Q:** How to customize the sentiment labels?  
  **A:** Modify the sentiment logic in `utils.py` or the prompt in `prompts.py`.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or support, contact:  
**Your Name**  
Email: your.email@example.com  
GitHub: https://github.com/yourusername

---

Thank you for using the AI-Powered Customer Feedback Summarizer!
