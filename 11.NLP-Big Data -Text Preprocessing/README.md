<h1> NLP Text Preprocessing Pipeline</h1>

<p>
Text is one of the most unstructured forms of data. Unlike numerical datasets,
raw text contains noise, inconsistent formats, unnecessary symbols, and different
word variations. Therefore, text preprocessing is an essential step to transform
raw text into clean and meaningful data suitable for Natural Language Processing (NLP)
tasks.
</p>


<hr>


<h2>1.  Project Overview</h2>

<p>
This project demonstrates a complete NLP text preprocessing pipeline including
text cleaning, tokenization, noise removal, stop-word filtering, case normalization,
and lexical normalization using stemming and lemmatization techniques.
</p>

<p>
The objective is to convert raw text documents into structured and analysis-ready
text representations for downstream NLP applications such as text classification,
sentiment analysis, and information extraction.
</p>


<hr>


<h2>2.  Objectives</h2>

<ul>

<li>
Clean raw text data by removing unnecessary characters and noise.
</li>

<li>
Convert text documents into individual tokens.
</li>

<li>
Remove stop words and non-informative words.
</li>

<li>
Normalize text representation using stemming and lemmatization.
</li>

<li>
Prepare processed text data for machine learning and NLP models.
</li>

</ul>


<hr>


<h2>3.  Text Preprocessing Workflow</h2>


<pre>

Raw Text Data

       ↓

Remove New Lines & Tabs

       ↓

Remove HTML Tags

       ↓

Remove URLs

       ↓

Remove Special Characters & Punctuation

       ↓

Tokenization

       ↓

Lowercase Conversion

       ↓

Stop Word Removal

       ↓

Stemming / Lemmatization

       ↓

Clean NLP Dataset

</pre>


<hr>


<h2>4.  Data Cleaning</h2>


<h3>a. Remove New Lines and Tabs</h3>

<p>
Raw text often contains unnecessary spaces, line breaks, and tab characters.
These are removed to create a consistent text format.
</p>


<pre>
clean_data = data.replace("\n", " ")

clean_data = clean_data.replace("\t", " ")

clean_data = re.sub(
    re.compile(r'\s+'),
    " ",
    clean_data)
</pre>



<h3>b. Remove Unicode and Non-Informative Characters</h3>

<p>
Non-semantic characters such as encoding symbols, special Unicode characters,
and unwanted formatting are removed.
</p>


<pre>
clean_data = clean_data.encode(
    "ascii",
    "ignore")

clean_data = clean_data.decode()
</pre>



<h3>c. Remove URLs</h3>

<p>
Web links do not provide meaningful information in many NLP tasks and are removed.
</p>


<pre>
clean_data = re.sub(
r'https?://[a-zA-Z0-9\.\/\-_?=;&]*',
'',
clean_data)
</pre>



<h3>d. Remove HTML Tags</h3>

<pre>
clean_data = re.sub(
r'<[^>]+>',
'',
clean_data)
</pre>



<h3>e. Remove Special Characters and Punctuation</h3>


<pre>

unwanted_punc = [
'"',"'",'=','@','&','%',
'.',',',':','\\','$',
'^','<','>','!',
'?','{','}',';',
'\n','\t']


for punc in unwanted_punc:
    clean_data = clean_data.replace(
        punc,
        "")

</pre>


<hr>


<h2>5.  Tokenization</h2>


<p>
Tokenization is the process of splitting text documents into smaller units
called tokens. Tokens can represent words, sentences, or phrases.
</p>


<p>
In this project, the NLTK library is used for word-level tokenization.
</p>

<pre>

from nltk.tokenize import word_tokenize
tokens = word_tokenize(clean_data)
print(tokens)

</pre>


<h3>Example:</h3>

<table border="1">

<tr>
<th>Before</th>
<th>After Tokenization</th>
</tr>

<tr>
<td>
"Machine learning is powerful"
</td>

<td>
["Machine","learning","is","powerful"]
</td>

</tr>

</table>


<hr>


<h2>6.  Stop Word Removal</h2>
<p>
Stop words are common words that provide limited information for NLP models.
Removing them helps models focus on meaningful words.
</p>


<p>
Examples:</p>

<ul>

<li>is</li>
<li>am</li>
<li>the</li>
<li>of</li>
<li>in</li>

</ul>


<pre>

from nltk.corpus import stopwords


stop_words = stopwords.words(
"english")

</pre>

<hr>


<h2>7.  Case Normalization</h2>


<p>
All words are converted into lowercase to avoid treating the same word as different
features.
</p>


<pre>

normal_tokens = []

for token in tokens:
    normal_tokens.append(
        token.lower())

</pre>


Example:

<table border="1">

<tr>
<th>Before</th>
<th>After</th>
</tr>

<tr>
<td>
Machine, MACHINE, machine
</td>

<td>
machine
</td>

</tr>

</table>


<hr>


<h2>8.  Lexicon Normalization</h2>


<p>
Words can have multiple forms with similar meanings.
Normalization converts different word variations into a common root form.
</p>


<p>
Two common methods:</p>


<ul>

<li>
<b>Stemming:</b> Removes word suffixes to obtain root-like forms.
</li>

<li>
<b>Lemmatization:</b> Converts words into their meaningful dictionary form.
</li>

</ul>

<h3>Stemming Example</h3>


<pre>

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmer.stem(
"believes")

</pre>


Output:

<pre>
believ
</pre>



<h3>Lemmatization Example</h3>

<pre>
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize(
"believes")

</pre>


Output:

<pre>
believe
</pre>

<hr>


<h2>9. 🛠 Technologies Used</h2>


<ul>

<li><b>Programming Language:</b> Python</li>
<li><b>NLP Library:</b> NLTK</li>
<li><b>Text Processing:</b> Regular Expressions (Regex)</li>
<li><b>Libraries:</b> Pandas, NumPy</li>
<li><b>Techniques:</b>
Tokenization, Stop-word Removal, Stemming, Lemmatization
</li>

</ul>
<hr>


<h2>10.  Key Insights</h2>

<ul>

<li>
Text preprocessing is a critical step because raw text contains significant noise.
</li>

<li>
Removing unnecessary characters improves model performance and reduces complexity.
</li>
<li>
Tokenization transforms unstructured text into machine-readable units.
</li>
<li>
Stop-word removal reduces irrelevant information.
</li>
<li>
Lemmatization provides better linguistic normalization compared with simple stemming.
</li>
<li>
A well-designed preprocessing pipeline improves the quality of downstream NLP models.
</li>

</ul>
<hr>


<h2> Conclusion</h2>

<p>
This project demonstrates a complete NLP preprocessing workflow that transforms
raw unstructured text into clean and normalized data. The pipeline prepares text
for advanced NLP applications including machine learning classification,
sentiment analysis, and large language model applications.
</p>