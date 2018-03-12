---
layout:     post
title:      Email Sentiment Analysis
date:       2015-01-22
summary:    Below is the result of a weekend playing around with doing sentiment analysis over my own corpus of personal email. Email is one of the often overlooked goldmines of user data, previous work at SendGrid and Return Path have given me some insight into this.
permalink: /email-sentiment-analysis/
---

<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>
<script src="http://code.highcharts.com/highcharts.js"></script>
<script src="http://code.highcharts.com/modules/exporting.js"></script>

Below is the result of a weekend playing around with doing sentiment analysis over my own corpus of personal email. Email is one of the often overlooked goldmines of user data, previous work at SendGrid and Return Path have given me some insight into this.

# First approach

First I tried the basic approach. Use Naive Bayes and bag-of-words over the [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/) to create our classifier. The graph below shows this analysis over the past few years of my email.

<div id="1" style="min-width: 310px; height: 400px; margin: 0 auto"></div>

Can you guess when I got married? 😄

This is a good sign. Already, and without much work, I can see clear points in the past where my email should've been happier. Let's also take a look at the volume of my email over this time:

<div id="2" style="min-width: 310px; height: 400px; margin: 0 auto"></div>

Good to know.

Looking back up at this graph and the one before though, one of the first thoughts I have looking at them are all the noisy emails in there. The marketing fluff. A simple way to remove this is ignoring all the email with a header `Precedence: Bulk`, high bulkscore, or the word bulk or bounce in the from or reply-to address, or had something like `*no*reply*@*` in there. Simple filtering, but seems to do a decent job, and it's as far as I'm willing to take it for now.

Now let's look at the volume again.

<div id="3" style="min-width: 310px; height: 400px; margin: 0 auto"></div>

There we go. Looking through my email, this definitely correlates more strongly with times I was receiving more personal emails. It's not perfect, but pretty close.

Now, let's try to improve my classifier. First thing I could do here is use [Tf–idf](https://en.wikipedia.org/wiki/Tf–idf).

<div id="4" style="min-width: 310px; height: 400px; margin: 0 auto"></div>

This looks better, and after giving the data a quick look over, these results are definitely more inline with the corpus. Now, let's try using a SVM and not NB.

<div id="5" style="min-width: 310px; height: 400px; margin: 0 auto"></div>

Awesome, even better. Although it may be hard to tell from the graphs, so far the F1 scores have agreed, we're improving. Oh, and now's probably a good time to mention how I'm tokenizing. Starting off with single words, though I plan to try [skipgrams](https://en.wikipedia.org/wiki/N-gram#Skip-Gram) at some point. The only punctuation I include are "!" and "?" -- and I try to use anything that looks like an emoji, :) :D ;-) etc. All this, plus using lemmas and removing stopwords.

``` python
def emoji(s):
  eyes, nose, mouth = [':', ';'], ['-'], ['D', ')', '(', '3']
  for i in xrange(len(s)-1):
    x = s[i:i+3]
    if x[0] in eyes and x[1] in nose and len(x) > 2 and x[2] in mouth:
      yield x
    elif x[0] in eyes and x[1] in mouth:
      yield x[:2]

def punctuation(s):
  meaningful_punctuation = ['!', '?']
  for x in s:
    if x in meaningful_punctuation:
      yield x

def tok(m):
  stops = set(stopwords.words("english"))
  az = re.sub("[^a-zA-Z]", " ", m).lower()
  words = [w.lemma for w in TextBlob(az).words if w not in stops]
  return words + list(punctuation(m)) + list(emoji(m))
```

# Second approach

Next, strongly influenced by [this work](http://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf) coming out of Standford's AI group, I decided to try and use unsupervised learning to build word vectors, then a supervised approach to categorize my corpus.

To build the word vectors, I used [word2vec](http://arxiv.org/pdf/1301.3781.pdf) within [gensim](http://radimrehurek.com/gensim/about.html), though there are many other implementations out there. A bit more about it on the authors blog [here](http://radimrehurek.com/2013/09/deep-learning-with-word2vec-and-gensim/), look around his blog a bit while you're there, the guy is brilliant. With word2vec I set the vector dimensionality at 700, had a minimum word count of 50, set the context window size to 12, and downsampled at 1e-3.

It can be an ***insane amount of fun*** to play around with the model once it's done. I definitely spent a few hours dicking around with it. Wicked cool stuff. Also, for other great papers on this subject, see [this guy](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), [this guy](http://www.puttypeg.net/papers/quantum-senses.pdf), and [this guy](http://www.aclweb.org/anthology/J/J98/J98-1004.pdf). It's also worth checking out [GloVe](http://nlp.stanford.edu/projects/glove/).

I used [wikipdea's corpus](http://dumps.wikimedia.org/enwiki/latest/), a roughly 60 gig xml file, and the [UMBC webbase corpus](http://ebiquity.umbc.edu/blogger/2013/05/01/umbc-webbase-corpus-of-3b-english-words/), about the same size. Altogether about 7 billion words. This took awhile. I also had to raise the stack size limit on my iMac, otherwise it'd never complete before the process would be kicked off. For those interested, you can rise your stack size to the hard limit using:

``` bash
adammenges@imac:~/Development/email-sentiment-analysis|master
$ ulimit -a
-t: cpu time (seconds)              unlimited
-f: file size (blocks)              unlimited
-d: data seg size (kbytes)          unlimited
-s: stack size (kbytes)             8192
-c: core file size (blocks)         0
-v: address space (kbytes)          unlimited
-l: locked-in-memory size (kbytes)  unlimited
-u: processes                       709
-n: file descriptors                2560
adammenges@imac:~/Development/email-sentiment-analysis|master
$ ulimit -s hard
adammenges@imac:~/Development/email-sentiment-analysis|master
$ ulimit -a
-t: cpu time (seconds)              unlimited
-f: file size (blocks)              unlimited
-d: data seg size (kbytes)          unlimited
-s: stack size (kbytes)             65532
-c: core file size (blocks)         0
-v: address space (kbytes)          unlimited
-l: locked-in-memory size (kbytes)  unlimited
-u: processes                       709
-n: file descriptors                2560
adammenges@imac:~/Development/email-sentiment-analysis|master
```

Now on to the supervised part. For this, I decided to continue with SVMs, with simple k-means clustering for my feature vectors. Again, starting with IMDBs dataset.

Also, this time, I'm going to throw in a corpus made using twitter. The approach I used can be found [here](http://deepthoughtinc.com/wp-content/uploads/2011/01/Twitter-as-a-Corpus-for-Sentiment-Analysis-and-Opinion-Mining.pdf). Pretty simple idea. Tweets that have happy emoji in them are most likely positive. Tweets with sad emoji in them are most likely negative. I take their work a step further and, if a tweet has a url in it, get the article text using readability, continuing along with their base assumption that positive emoji means good and negative emoji means bad. There must be cases where this doesn't hold, but I haven't seen any yet. There is good information to glean here.

Movie reviews are definitely written with a certain tone, combining these datasets will help overcome some of that by throwing in more common language, albeit some of that is twitter language.

I'd kill for Facebook's dataset. Not only would it be a fairly good representation of how people normally talk to their friends, their users, as of recently, even have the option to label a post with with it's 'Feeling'. I *wonder* why they added this? 😏

All this took a little while, especially without a good cuda capable gpu. Go grab coffee, or perhaps while you're waiting, read more of their paper, or any of the others I linked to here.

The results are very promising:

<div id="7" style="min-width: 310px; height: 400px; margin: 0 auto"></div>

Looking closely at these results, and over my own email, these are by far the best. Getting married, my job at Apple, graduating college, and any other events in my life that have resulted in congratulations from friends and family members over email are clearly shown.

The 'qualified self' is very interesting to me. Email is a great place to start. I've also got all my IRC logs since the beginning of time. And now that Messages on OSX included SMSs, of course along with the iMessages, in a few years I could have a corpus of chats to analyze. Facebook Messenger and others also make it easy to fetch chats from. I'm super interested in this. Texts likely say a whole lot more about my mood over time then email does, as the majority of my remote communication happens there. For those interested, something like this below could be used collect it. Along these same lines, [Stephen Wolfram's Personal Analytics](http://blog.stephenwolfram.com/2012/03/the-personal-analytics-of-my-life/) is a good read, and I'd watch a few of [these](http://vimeo.com/groups/quantifiedself) as well.

``` bash
on write_to_file(this_data, event_description, target_file)
  set timeStr to time string of (current date)
  do shell script "echo  " & quoted form of timeStr & " >>  " & quoted form of target_file
  do shell script "echo  " & quoted form of event_description & " >>  " & quoted form of target_file
  do shell script "echo  " & quoted form of this_data & " >>  " & quoted form of target_file
end write_to_file

using terms from application "Messages"
  on message sent theMessage with eventDescription for theChat
    my write_to_file(theMessage, eventDescription, "/Users/adammenges/corpora/messages/" & theChat) -- TODO: pull name from theChat
  end message sent

  on message received theMessage from theBuddy with eventDescription for theChat
    my write_to_file(theMessage, eventDescription, "/Users/adammenges/corpora/messages/" & theChat)
  end message received
end using terms from
```

# Ending thoughts

Next I'm going to try some 1D CNN approaches. There are also other neat things you could try to glean here. With whom do I have the strongest relationships? The worst? What kind of things am I interested in? From there, what's my mean purchase amount? My salary? Where have I traveled to? I wonder what forecasting you could do. Where am I most likely to travel to next? What am I most likely to buy?

For those interested in seeing what all can be done here, first grab all the corpora linked to here, download your own email using something like this below, pull up ipython, `import sklearn`, and have fun!

``` python
def corpus(username, password, server='imap.gmail.com', mailbox="[Gmail]/All Mail"):
  import imaplib
  import os
  mail = imaplib.IMAP4_SSL(server)
  mail.login(username, password)
  mail.select(mailbox)
  result, data = mail.search(None, 'ALL')
  directory = 'corpus-' + username

  if not os.path.exists(directory):
    os.makedirs(directory)

  for x in data[0].split():
    typ, email = mail.fetch(x, '(RFC822)')
    with open(directory + '/' + x, 'w') as f:
      f.write(email[0][1])

def parse(message):
  from bs4 import BeautifulSoup
  import email
  import dateutil.parser
  msg = email.message_from_string(message)
  lhs = {}
  lhs['Date'] = dateutil.parser.parse(msg['Date'], fuzzy=True).strftime('%Y-%m-%d')

  for part in msg.walk():
    if part.get_content_type() == 'text/plain':
      lhs['Body'] = BeautifulSoup(part.get_payload()).get_text()

  return lhs
```

I'll spend the next few weekends playing around with this, try some other deep approaches, but I've nonetheless been impressed by these results. If you'd like to chat, or try out anything interesting here, be sure to [get in touch](https://blog.adammenges.com/about/). Maybe I'll swap out the Tf–idf for a Recurrent Neural Network, or perhaps see how Dato or Indico preform, or try out [NB-SVM](http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf). I'm excited to see where NLP and Machine Learning head to next, to be a part of it and contribute. Machine learning is definitely one of, if not *the most*, exciting place to be working. We're at the cusp of a huge incoming shift. These technologies will have profound impact on our society, with predictions ranging from utopic to apocalyptic.

> “Look at you, hacker: a pathetic creature of meat and bone, panting and sweating as you run through my corridors. How can you challenge a perfect, immortal machine?”
> ― Ken Levine

<script type="text/javascript">
  $('#1').highcharts({
      chart: {
          zoomType: 'x'
      },
      title: {
          text: ''
      },
      subtitle: {
          text: document.ontouchstart === undefined ?
                  'Click and drag in the plot area to zoom in' :
                  'Pinch the chart to zoom in'
      },
      xAxis: {
          type: 'datetime',
          minRange: 14 * 24 * 3600000 // fourteen days
      },
      yAxis: {
          title: {
              text: 'Happiness'
          }
      },
      legend: {
          enabled: false
      },
      plotOptions: {
          area: {
              fillColor: {
                  linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1},
                  stops: [
                      [0, Highcharts.getOptions().colors[0]],
                      [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
                  ]
              },
              marker: {
                  radius: 2
              },
              lineWidth: 1,
              states: {
                  hover: {
                      lineWidth: 1
                  }
              },
              threshold: null
          }
      },
      series: [{
          type: 'area',
          name: '',
          pointInterval: 24 * 3600 * 1000,
          pointStart: Date.UTC(2010, 0, 0),
          data: [12, 2, 13, 24, 25, 10, 14, 12, 12, 8, 6, 0, 22, 4, 1, 17, 24, 23, 15, 2, 17, 26, 2, 25, 3, 3, 0, 20, 22, 20, 22, 18, 17, 12, 27, 7, 25, 25, 21, 8, 18, 0, 21, 18, 15, 25, 2, 23, 24, 21, 8, 10, 12, 6, 11, 18, 8, 14, 26, 1, 22, 18, 15, 25, 7, 25, 23, 2, 26, 22, 12, 1, 15, 27, 5, 23, 1, 13, 24, 0, 9, 24, 14, 6, 24, 11, 14, 20, 13, 6, 27, 13, 11, 17, 13, 14, 33, 17, 19, 9, 20, 27, 12, 20, 1, 9, 8, 24, 23, 15, 11, 21, 21, 26, 0, 16, 7, 1, 0, 24, 26, 17, 10, 6, 23, 29, 18, 7, 8, 24, 16, 19, 19, 13, 15, 2, 6, 16, 17, 2, 26, 16, 27, 9, 15, 14, 19, 12, 11, 3, 17, 25, 26, 5, 12, 10, 14, 17, 21, 26, 3, 1, 10, 17, 7, 7, 23, 14, 9, 8, 16, 3, 27, 9, 14, 20, 1, 12, 8, 19, 1, 2, 29, 16, 16, 1, 25, 6, 15, 26, 21, 3, 7, 11, 26, 12, 27, 16, 8, 10, 27, 7, 13, 27, 11, 22, 16, 3, 3, 10, 23, 10, 19, 4, 2, 20, 26, 18, 0, 13, 27, 13, 22, 12, 0, 17, 3, 25, 26, 2, 20, 10, 6, 7, 25, 13, 23, 10, 12, 6, 9, 1, 12, 13, 20, 13, 8, 16, 23, 25, 26, 18, 11, 3, 26, 16, 21, 4, 27, 3, 23, 7, 10, 8, 8, 13, 18, 27, 26, 0, 21, 7, 11, 6, 0, 22, 7, 20, 11, 16, 5, 2, 24, 2, 2, 25, 0, 9, 8, 19, 14, 2, 12, 25, 1, 22, 13, 44, 17, 38, 16, 14, 18, 1, 7, 3, 24, 26, 12, 15, 21, 10, 0, 22, 18, 25, 7, 9, 2, 26, 27, 26, 20, 2, 4, 4, 3, 21, 12, 1, 25, 19, 8, 24, 23, 9, 3, 26, 4, 6, 2, 8, 22, 8, 13, 6, 1, 3, 16, 23, 9, 7, 0, 18, 6, 12, 19, 26, 8, 3, 3, 0, 27, 30, 0, 25, 24, 0, 17, 11, 10, 26, 25, 9, 15, 1, 22, 21, 17, 9, 0, 15, 8, 9, 14, 26, 25, 6, 9, 4, 17, 25, 13, 26, 1, 24, 1, 14, 23, 4, 17, 18, 21, 27, 8, 12, 26, 27, 4, 1, 11, 16, 0, 18, 21, 24, 19, 27, 18, 25, 24, 29, 44, 16, 2, 13, 6, 9, 10, 24, 12, 14, 12, 25, 13, 11, 23, 8, 15, 10, 2, 7, 7, 24, 0, 7, 9, 6, 13, 3, 4, 44, 22, 23, 6, 15, 26, 4, 4, 22, 7, 19, 12, 20, 6, 22, 18, 11, 26, 0, 6, 17, 15, 21, 19, 17, 13, 26, 23, 18, 13, 7, 7, 0, 22, 7, 25, 18, 1, 2, 12, 7, 0, 13, 26, 22, 21, 23, 2, 11, 21, 22, 15, 21, 4, 15, 10, 11, 21, 25, 10, 10, 8, 24, 9, 17, 26, 10, 1, 16, 2, 0, 15, 27, 8, 8, 6, 7, 24, 5, 25, 12, 15, 1, 13, 6, 2, 16, 20, 11, 24, 17, 10, 21, 22, 11, 3, 22, 14, 2, 10, 22, 27, 49, 43, 52, 55, 82, 10, 12, 12, 1, 27, 17, 57, 76, 37, 52, 61, 64, 59, 53, 50, 42, 46, 55, 85, 61, 67, 64, 65, 71, 38, 37, 23, 34, 44, 95, 13, 17, 4, 1, 25, 2, 5, 24, 11, 6, 2, 11, 25, 13, 25, 26, 5, 22, 2, 21, 13, 16, 14, 5, 11, 5, 6, 8, 25, 17, 27, 36, 26, 33, 46, 53, 13, 4, 2, 14, 24, 5, 1, 27, 23, 8, 8, 10, 6, 9, 20, 11, 0, 17, 3, 26, 15, 21, 8, 16, 11, 22, 5, 5, 1, 10, 26, 10, 20, 16, 12, 10, 6, 12, 12, 17, 14, 26, 15, 6, 20, 18, 3, 6, 12, 17, 7, 6, 19, 17, 3, 21, 12, 14, 20, 11, 17, 10, 8, 6, 21, 23, 9, 7, 1, 25, 8, 11, 5, 3, 0, 21, 2, 25, 24, 8, 12, 23, 2, 5, 3, 24, 26, 4, 26, 3, 0, 3, 3, 1, 24, 20, 2, 21, 10, 13, 3, 16, 2, 0, 19, 26, 21, 7, 16, 16, 13, 5, 4, 17, 24, 16, 17, 25, 27, 26, 24, 1, 23, 1, 25, 23, 11, 24, 13, 8, 21, 16, 16, 2, 1, 10, 10, 2, 2, 9, 22, 12, 21, 22, 0, 4, 20, 16, 16, 1, 15, 26, 5, 12, 23, 25, 24, 14, 2, 6, 11, 11, 18, 10, 19, 8, 23, 10, 1, 14, 27, 2, 6, 9, 14, 13, 10, 22, 13, 20, 18, 20, 12, 24, 10, 26, 24, 22, 9, 21, 4, 19, 13, 24, 24, 8, 12, 11, 24, 4, 27, 10, 27, 19, 17, 20, 26, 16, 9, 22, 24, 21, 25, 0, 25, 23, 6, 25, 5, 27, 3, 20, 16, 15, 27, 1, 11, 13, 21, 22, 4, 1, 19, 20, 7, 13, 3, 10, 2, 27, 5, 4, 24, 1, 13, 1, 21, 14, 19, 21, 6, 16, 23, 7, 5, 26, 20, 15, 27, 17, 6, 9, 16, 17, 16, 19, 26, 25, 13, 16, 0, 21, 5, 12, 19, 14, 2, 13, 0, 8, 5, 6, 26, 12, 11, 3, 7, 13, 15, 15, 17, 8, 13, 16, 16, 17, 9, 0, 11, 17, 24, 6, 2, 24, 8, 1, 33, 42, 24, 29, 15, 9, 2, 25, 6, 13, 19, 18, 14, 17, 27, 16, 6, 8, 3, 6, 20, 3, 5, 1, 6, 14, 7, 8, 5, 5, 2, 2, 7, 9, 17, 8, 20, 16, 18, 1, 2, 13, 3, 14, 36, 2, 15, 18, 2, 24, 9, 23, 3, 8, 18, 18, 13, 0, 10, 20, 23, 26, 3, 11, 23, 13, 25, 6, 7, 2, 8, 19, 9, 21, 18, 25, 9, 27, 4, 8, 15, 24, 20, 23, 9, 23, 0, 0, 17, 20, 5, 15, 27, 6, 1, 7, 19, 12, 12, 25, 16, 18, 17, 26, 19, 4, 18, 19, 27, 9, 16, 4, 23, 4, 13, 13, 0, 7, 7, 14, 24, 6, 26, 27, 20, 14, 26, 7, 6, 1, 25, 22, 24, 25, 7, 17, 4, 21, 12, 11, 21, 20, 23, 14, 21, 14, 22, 17, 15, 22, 2, 26, 24, 12, 6, 15, 20, 13, 19, 0, 19, 25, 4, 13, 5, 4, 8, 22, 20, 27, 12, 25, 2, 25, 11, 13, 3, 7, 3, 2, 10, 16, 2, 21, 13, 16, 6, 17, 9, 9, 14, 4, 12, 25, 24, 2, 23, 23, 7, 17, 7, 18, 19, 8, 27, 18, 18, 14, 18, 6, 4, 0, 12, 11, 4, 19, 2, 7, 6, 2, 1, 17, 20, 12, 13, 12, 6, 25, 4, 8, 18, 8, 13, 27, 0, 8, 12, 8, 22, 17, 16, 9, 1, 16, 16, 17, 9, 7, 7, 0, 21, 18, 23, 16, 10, 1, 20, 16, 24, 16, 12, 24, 17, 11, 21, 24, 5, 3, 27, 20, 2, 0, 17, 0, 14, 11, 27, 20, 3, 11, 12, 12, 9, 13, 5, 10, 0, 11, 14, 51, 0, 19, 13, 13, 4, 25, 10, 15, 7, 27, 18, 14, 42, 16, 18, 16, 27, 10, 13, 0, 1, 25, 1, 9, 23, 1, 27, 26, 19, 22, 15, 27, 13, 26, 17, 12, 19, 9, 37, 38, 29, 54, 48, 9, 26, 20, 8, 18, 26, 11, 9, 23, 7, 2, 7, 13, 24, 27, 2, 10, 19, 5, 7, 4, 1, 7, 1, 20, 16, 14, 8, 9, 9, 4, 8, 9, 21, 24, 24, 22, 20, 27, 23, 16, 4, 10, 6, 7, 21, 18, 8, 20, 10, 9, 12, 8, 14, 1, 24, 13, 24, 10, 4, 13, 20, 8, 24, 10, 15, 16, 15, 9, 2, 18, 23, 24, 22, 22, 5, 20, 26, 27, 16, 9, 12, 4, 16, 1, 0, 3, 6, 10, 20, 6, 7, 26, 23, 30, 18, 20, 18, 2, 23, 18, 20, 10, 20, 24, 20, 19, 16, 19, 4, 12, 5, 10, 9, 9, 27, 17, 9, 19, 19, 11, 26, 24, 20, 7, 7, 0, 4, 11, 18, 9, 24, 12, 19, 8, 26, 24, 23, 12, 24, 8, 4, 6, 6, 6, 6, 1, 7, 0, 6, 3, 23, 24, 16, 10, 9, 22, 16, 23, 7, 25, 20, 17, 22, 11, 14, 5, 8, 22, 7, 24, 24, 26, 10, 0, 25, 10, 18, 12, 26, 14, 3, 14, 10, 18, 22, 27, 20, 4, 14, 6, 19, 26, 6, 10, 12, 42, 18, 19, 15, 5, 7, 29, 26, 6, 15, 24, 6, 15, 24, 8, 20, 7, 7, 25, 10, 9, 27, 13, 5, 28, 3, 21, 6, 14, 27, 7, 22, 3, 27, 14, 26, 21, 3, 19, 18, 11, 16, 7, 6, 23, 13, 13, 20, 4, 5, 22, 10, 7, 14, 17, 29, 11, 22, 27, 12, 4, 23, 11, 4, 11, 11, 12, 16, 10, 11, 23, 29, 3, 10, 17, 14, 7, 28, 25, 25, 11, 8, 4, 25, 29, 8, 21, 20, 23, 18, 19, 24, 27, 9, 24, 18, 25, 19, 6, 28, 16, 5, 13, 8, 15, 20, 28, 13, 25, 12, 8, 7, 28, 21, 15, 3, 23, 3, 26, 28, 21, 15, 25, 4, 18, 24, 3, 24, 11, 19, 26, 11, 22, 9, 21, 3, 7, 12, 23, 16, 5, 5, 12, 19, 22, 15, 11, 23, 19, 20, 22, 22, 10, 27, 6, 25, 25, 28, 26, 4, 20, 18, 13, 16, 4, 22, 29, 19, 6, 12, 3, 13, 20, 24, 22, 19, 11, 9, 16, 28, 19, 29, 27, 25, 7, 5, 7, 13, 13, 7, 23, 29, 25, 17, 23, 5, 19, 15, 23, 3, 24, 10, 7, 6, 15, 16, 28, 12, 27, 29, 13, 27, 28, 23, 10, 16, 9, 11, 5, 20, 3, 3, 24, 29, 12, 26, 10, 25, 19, 6, 6, 29, 18, 9, 9, 29, 15, 9, 19, 25, 6, 11, 19, 3, 7, 29, 16, 14, 20, 16, 6, 28, 28, 9, 22, 21, 8, 27, 22, 6, 4, 29, 24, 15, 3, 6, 16, 3, 28, 11, 20, 4, 15, 21, 16, 14, 11, 12, 3, 17, 7, 17, 20, 4, 5, 21, 25, 16, 18, 8, 23, 15, 21, 13, 15, 8, 17, 11, 19, 11, 26, 11, 18, 21, 19, 17, 3, 22, 4, 21, 25, 23, 24, 24, 17, 5, 6, 25, 23, 29, 28, 29, 14, 10, 19, 3, 19, 13, 12, 19, 10, 6, 13, 13, 9, 9, 10, 4, 14, 5, 13, 17, 18, 12, 10, 6, 11, 14, 22, 6, 29, 9, 16, 27, 8, 20, 14, 19, 3, 14, 25, 11, 17, 3, 15, 18, 33, 54]
      }]
  });
  $('#2').highcharts({
      chart: {
          zoomType: 'x'
      },
      title: {
          text: ''
      },
      subtitle: {
          text: document.ontouchstart === undefined ?
                  'Click and drag in the plot area to zoom in' :
                  'Pinch the chart to zoom in'
      },
      xAxis: {
          type: 'datetime',
          minRange: 14 * 24 * 3600000 // fourteen days
      },
      yAxis: {
          title: {
              text: 'Volume'
          }
      },
      legend: {
          enabled: false
      },
      plotOptions: {
          area: {
              fillColor: {
                  linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1},
                  stops: [
                      [0, Highcharts.getOptions().colors[0]],
                      [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
                  ]
              },
              marker: {
                  radius: 2
              },
              lineWidth: 1,
              states: {
                  hover: {
                      lineWidth: 1
                  }
              },
              threshold: null
          }
      },

      series: [{
          type: 'area',
          name: '',
          pointInterval: 24 * 3600 * 1000,
          pointStart: Date.UTC(2010, 0, 0),
          data: [67, 47, 44, 59, 53, 66, 69, 50, 68, 67, 49, 46, 55, 50, 45, 63, 53, 60, 46, 45, 66, 57, 55, 46, 53, 52, 65, 51, 50, 56, 53, 60, 47, 63, 48, 50, 67, 67, 62, 61, 61, 46, 61, 55, 49, 65, 66, 44, 56, 56, 54, 45, 61, 48, 68, 55, 59, 45, 5, 67, 53, 46, 61, 64, 45, 51, 58, 52, 45, 52, 53, 54, 51, 51, 64, 60, 51, 50, 44, 57, 52, 58, 53, 56, 51, 65, 57, 49, 62, 55, 53, 59, 46, 68, 51, 48, 45, 69, 54, 46, 58, 45, 47, 50, 53, 52, 44, 51, 66, 54, 51, 66, 69, 47, 48, 54, 54, 68, 60, 62, 54, 51, 66, 57, 50, 49, 69, 52, 67, 62, 47, 57, 47, 63, 63, 49, 45, 64, 64, 68, 65, 51, 58, 48, 58, 48, 46, 59, 44, 58, 59, 56, 47, 50, 52, 65, 48, 51, 56, 59, 59, 69, 44, 44, 44, 58, 54, 55, 68, 49, 55, 55, 68, 57, 46, 64, 62, 51, 60, 66, 48, 64, 44, 59, 44, 45, 60, 64, 69, 56, 56, 51, 60, 69, 57, 60, 55, 46, 53, 59, 45, 44, 48, 55, 49, 59, 65, 57, 56, 46, 52, 46, 47, 48, 45, 68, 60, 47, 66, 51, 53, 47, 49, 48, 48, 52, 69, 59, 59, 44, 65, 1, 58, 62, 55, 58, 54, 52, 69, 57, 58, 63, 67, 67, 59, 47, 61, 62, 45, 49, 45, 62, 61, 63, 54, 67, 55, 68, 48, 59, 57, 61, 65, 53, 65, 50, 56, 60, 60, 53, 58, 52, 55, 54, 55, 66, 66, 55, 44, 52, 47, 68, 50, 61, 53, 44, 63, 67, 44, 52, 46, 49, 67, 66, 46, 51, 59, 47, 44, 63, 49, 60, 68, 59, 48, 53, 54, 58, 57, 53, 69, 50, 62, 68, 50, 44, 45, 66, 54, 54, 58, 65, 69, 48, 47, 68, 64, 55, 67, 66, 50, 55, 52, 59, 65, 52, 58, 66, 60, 45, 66, 65, 66, 47, 47, 62, 66, 46, 69, 57, 50, 58, 50, 48, 52, 59, 67, 62, 51, 51, 46, 55, 55, 6, 63, 49, 50, 47, 48, 61, 54, 46, 62, 50, 56, 60, 62, 57, 44, 45, 53, 47, 67, 52, 69, 67, 59, 65, 63, 52, 53, 51, 45, 50, 68, 47, 69, 67, 44, 62, 47, 69, 66, 44, 49, 63, 48, 46, 54, 50, 47, 62, 64, 61, 51, 67, 57, 50, 50, 48, 64, 53, 64, 66, 59, 48, 65, 51, 63, 47, 65, 48, 67, 48, 49, 57, 64, 54, 45, 60, 66, 50, 50, 59, 51, 68, 61, 58, 50, 67, 49, 54, 44, 68, 57, 46, 62, 69, 45, 66, 52, 66, 49, 67, 55, 60, 51, 52, 53, 62, 48, 67, 68, 1, 48, 60, 65, 51, 65, 56, 52, 59, 52, 57, 57, 53, 63, 59, 55, 58, 53, 55, 55, 54, 51, 45, 69, 53, 49, 48, 46, 56, 50, 56, 51, 64, 65, 59, 58, 45, 63, 67, 63, 64, 64, 64, 45, 51, 60, 56, 60, 52, 66, 44, 55, 58, 58, 48, 51, 49, 53, 67, 62, 57, 60, 45, 45, 60, 58, 47, 44, 46, 60, 54, 59, 63, 57, 52, 54, 53, 57, 50, 44, 45, 54, 64, 66, 69, 58, 61, 62, 66, 52, 46, 62, 58, 44, 56, 63, 48, 64, 69, 69, 56, 64, 59, 48, 65, 68, 52, 53, 55, 52, 0, 44, 63, 67, 53, 47, 51, 50, 52, 58, 60, 68, 56, 52, 49, 48, 56, 51, 55, 49, 57, 61, 61, 58, 50, 48, 62, 57, 60, 45, 46, 56, 57, 66, 45, 53, 65, 47, 53, 54, 44, 64, 58, 52, 68, 48, 54, 63, 63, 59, 68, 55, 47, 67, 68, 64, 49, 59, 56, 64, 62, 61, 63, 63, 46, 57, 46, 54, 49, 60, 51, 44, 64, 53, 62, 58, 59, 60, 56, 47, 46, 50, 56, 68, 57, 56, 63, 51, 65, 68, 58, 63, 57, 48, 66, 64, 59, 68, 46, 69, 57, 60, 54, 53, 69, 60, 68, 53, 57, 69, 46, 48, 65, 56, 47, 51, 52, 64, 55, 66, 50, 64, 56, 68, 60, 61, 48, 54, 55, 68, 55, 48, 64, 50, 55, 50, 68, 49, 47, 46, 61, 50, 45, 63, 58, 47, 60, 57, 68, 46, 57, 55, 55, 62, 52, 65, 65, 68, 49, 54, 63, 54, 62, 50, 61, 66, 54, 56, 57, 67, 56, 61, 69, 55, 50, 57, 66, 56, 57, 64, 68, 46, 57, 66, 47, 54, 55, 51, 65, 51, 45, 52, 57, 8, 47, 65, 45, 45, 47, 48, 64, 56, 57, 63, 49, 49, 47, 48, 50, 45, 45, 66, 59, 63, 66, 46, 46, 46, 51, 65, 49, 69, 46, 62, 67, 46, 64, 67, 54, 67, 53, 62, 65, 46, 62, 51, 46, 67, 64, 44, 53, 64, 58, 69, 56, 63, 45, 45, 49, 53, 61, 59, 51, 61, 45, 53, 46, 53, 45, 57, 68, 56, 46, 0, 46, 48, 65, 55, 62, 57, 57, 64, 54, 50, 45, 68, 58, 58, 44, 46, 47, 51, 63, 52, 55, 60, 50, 62, 49, 69, 69, 48, 52, 54, 61, 61, 60, 45, 58, 55, 59, 46, 59, 63, 59, 62, 54, 51, 67, 45, 49, 55, 64, 51, 60, 62, 44, 57, 49, 62, 60, 47, 68, 68, 65, 58, 55, 53, 57, 56, 55, 62, 59, 57, 44, 67, 58, 47, 44, 67, 59, 47, 55, 58, 49, 69, 68, 63, 64, 5, 44, 62, 59, 58, 53, 46, 69, 69, 53, 49, 49, 61, 49, 50, 57, 47, 66, 49, 54, 48, 65, 53, 53, 50, 57, 2, 59, 66, 47, 67, 59, 48, 55, 59, 61, 57, 58, 47, 50, 47, 60, 68, 56, 46, 54, 61, 65, 45, 46, 53, 66, 1, 51, 46, 52, 66, 52, 48, 48, 59, 55, 48, 59, 48, 52, 69, 63, 66, 66, 65, 68, 51, 58, 47, 46, 59, 48, 47, 62, 67, 48, 60, 61, 64, 67, 68, 68, 62, 45, 44, 44, 53, 64, 69, 64, 55, 62, 50, 65, 64, 62, 62, 60, 57, 49, 61, 56, 47, 65, 5, 59, 65, 51, 52, 62, 60, 48, 62, 44, 55, 69, 55, 54, 67, 47, 61, 66, 65, 44, 60, 48, 49, 49, 65, 67, 44, 59, 48, 49, 44, 64, 52, 9, 63, 69, 46, 64, 47, 67, 63, 51, 58, 46, 59, 52, 56, 45, 61, 61, 59, 67, 64, 60, 49, 66, 54, 54, 57, 2, 62, 66, 55, 56, 50, 49, 69, 53, 44, 54, 50, 66, 49, 64, 68, 65, 60, 52, 54, 56, 63, 64, 63, 68, 49, 50, 50, 48, 60, 44, 65, 47, 50, 66, 44, 60, 53, 58, 66, 50, 57, 48, 68, 9, 67, 55, 57, 58, 46, 46, 68, 54, 65, 49, 45, 63, 58, 53, 57, 62, 44, 58, 52, 61, 64, 65, 62, 57, 55, 59, 60, 46, 48, 67, 50, 61, 56, 64, 56, 65, 46, 52, 48, 48, 50, 54, 55, 63, 46, 49, 64, 45, 61, 66, 48, 67, 50, 46, 67, 56, 58, 54, 67, 56, 52, 65, 51, 51, 50, 64, 55, 57, 66, 65, 69, 65, 49, 58, 49, 47, 49, 65, 45, 61, 45, 69, 62, 55, 67, 52, 58, 60, 54, 60, 55, 50, 63, 53, 52, 64, 58, 60, 48, 53, 51, 47, 55, 56, 65, 68, 66, 56, 48, 46, 44, 3, 58, 51, 64, 56, 66, 50, 45, 45, 61, 52, 51, 49, 66, 57, 65, 69, 55, 65, 50, 57, 57, 50, 62, 59, 8, 0, 0, 4, 50, 64, 69, 50, 52, 54, 56, 64, 58, 65, 47, 44, 53, 64, 44, 62, 44, 45, 51, 54, 61, 59, 54, 60, 4, 54, 63, 57, 63, 59, 44, 52, 65, 66, 57, 62, 53, 69, 69, 65, 57, 63, 60, 63, 51, 51, 65, 60, 64, 55, 53, 59, 60, 8, 8, 51, 57, 62, 56, 67, 68, 60, 54, 55, 58, 47, 54, 59, 48, 65, 48, 57, 44, 63, 61, 51, 67, 48, 47, 63, 67, 54, 7, 62, 50, 54, 56, 60, 46, 50, 67, 45, 53, 61, 61, 67, 51, 58, 51, 55, 60, 47, 66, 63, 69, 66, 62, 60, 58, 61, 45, 52, 60, 64, 60, 49, 62, 48, 68, 52, 63, 47, 44, 58, 49, 69, 66, 45, 58, 62, 56, 69, 68, 52, 53, 63, 53, 9, 62, 48, 60, 63, 56, 57, 53, 51, 50, 60, 68, 65, 46, 62, 69, 58, 60, 52, 50, 58, 52, 59, 68, 51, 46, 66, 54, 59, 68, 54, 67, 51, 62, 6, 59, 56, 57, 54, 69, 45, 53, 59, 66, 44, 57, 52, 49, 48, 69, 53, 66, 3, 66, 48, 65, 63, 64, 53, 51, 63, 57, 54, 59, 45, 44, 48, 68, 64, 58, 69, 52, 57, 56, 56, 62, 54, 57, 60, 62, 45, 62, 48, 63, 48, 49, 53, 60, 3, 62, 48, 56, 44, 58, 49, 62, 47, 64, 45, 48, 66, 58, 64, 48, 63, 52, 62, 68, 61, 57, 52, 63, 67, 6, 67, 45, 61, 69, 56, 63, 49, 51, 68, 54, 62, 49, 52, 67, 58, 62, 67, 45, 62, 59, 49, 66, 44, 62, 66, 5, 59, 51, 47, 52, 64, 67, 53, 51, 69, 48, 55, 55, 61, 63, 46, 49, 63, 62, 49, 50, 60, 45, 64, 56, 52, 4, 65, 67, 3, 0, 67, 49, 63, 57, 68, 67, 55, 63, 57, 52, 59, 52, 51, 66, 61, 68, 59, 52, 54, 48, 61, 58, 57, 66, 59, 2, 58, 68, 44, 64, 48, 52, 63, 63, 54, 47, 57, 56, 56, 53, 63, 57, 67, 65, 44, 65, 69, 58, 60, 68, 51, 63, 2, 9, 52, 60, 48, 57, 51, 44, 55, 65, 52, 64, 68, 50, 60, 63, 50, 62, 49, 54, 68, 49, 57, 44, 53, 50, 67, 65, 2, 58, 61, 55, 53, 61, 59, 56, 60, 66, 56, 52, 45, 45, 45, 46, 66, 64, 49, 63, 47, 47, 65, 69, 46, 54, 65, 1, 67, 48, 49, 60, 60, 59, 67, 52, 65, 60, 62, 56, 69, 69, 51, 51, 62, 64, 51, 63, 50, 65, 51, 46, 46, 53, 51, 66, 63, 62, 61, 63, 53, 45, 51, 54, 59, 45, 65, 62, 53, 64, 51, 61, 64, 56, 68, 48, 67, 69, 62, 65, 58, 65, 55, 49, 46, 54, 60, 46, 67, 53, 58, 55, 53, 58, 54, 45, 50, 54, 69, 44, 45, 63, 46, 55, 63, 49, 59, 48, 63, 51, 55, 67, 58, 66, 56, 60, 57, 66, 45, 66, 62, 49, 47, 60, 68, 59, 52, 57, 58, 46, 50, 57, 67, 61, 53, 47, 49, 57, 53, 54, 53, 60, 57, 44, 62, 58, 60, 67, 48, 64, 67, 56, 22, 10]
      }]
  });

$('#3').highcharts({
      chart: {
          zoomType: 'x'
      },
      title: {
          text: ''
      },
      subtitle: {
          text: document.ontouchstart === undefined ?
                  'Click and drag in the plot area to zoom in' :
                  'Pinch the chart to zoom in'
      },
      xAxis: {
          type: 'datetime',
          minRange: 14 * 24 * 3600000 // fourteen days
      },
      yAxis: {
          title: {
              text: 'Volume'
          }
      },
      legend: {
          enabled: false
      },
      plotOptions: {
          area: {
              fillColor: {
                  linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1},
                  stops: [
                      [0, Highcharts.getOptions().colors[0]],
                      [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
                  ]
              },
              marker: {
                  radius: 2
              },
              lineWidth: 1,
              states: {
                  hover: {
                      lineWidth: 1
                  }
              },
              threshold: null
          }
      },

      series: [{
          type: 'area',
          name: '',
          pointInterval: 24 * 3600 * 1000,
          pointStart: Date.UTC(2010, 0, 0),
          data: [39, 20, 17, 30, 22, 30, 26, 27, 34, 36, 22, 21, 18, 23, 15, 24, 24, 20, 20, 27, 30, 18, 21, 15, 27, 17, 40, 20, 30, 18, 20, 22, 21, 24, 23, 19, 35, 37, 26, 20, 19, 23, 25, 23, 18, 28, 29, 18, 34, 22, 23, 22, 29, 18, 22, 28, 25, 18, 4, 23, 27, 24, 27, 28, 12, 19, 27, 21, 20, 21, 26, 21, 23, 32, 23, 18, 28, 26, 14, 34, 24, 27, 21, 25, 15, 40, 27, 23, 33, 18, 25, 26, 12, 26, 23, 18, 15, 25, 28, 25, 22, 20, 18, 26, 29, 23, 20, 23, 26, 29, 21, 38, 33, 19, 21, 21, 24, 30, 30, 22, 26, 21, 23, 24, 20, 27, 40, 29, 35, 27, 15, 30, 24, 34, 20, 20, 15, 30, 27, 37, 23, 20, 34, 17, 21, 26, 15, 32, 20, 27, 27, 20, 24, 21, 23, 34, 18, 25, 21, 26, 30, 23, 17, 21, 21, 29, 19, 29, 22, 23, 24, 20, 26, 20, 22, 30, 24, 15, 31, 22, 14, 37, 12, 30, 17, 16, 23, 41, 28, 29, 28, 20, 24, 34, 24, 31, 24, 15, 25, 20, 18, 18, 17, 20, 24, 18, 29, 20, 26, 19, 25, 22, 12, 15, 16, 38, 29, 19, 33, 26, 22, 16, 14, 19, 27, 28, 26, 23, 31, 14, 29, 6, 32, 29, 21, 28, 28, 27, 35, 26, 31, 26, 35, 32, 26, 18, 20, 32, 12, 19, 23, 26, 36, 30, 18, 34, 34, 21, 14, 27, 24, 27, 28, 16, 36, 25, 23, 34, 33, 32, 38, 14, 33, 24, 26, 31, 35, 21, 10, 14, 16, 34, 28, 20, 24, 14, 30, 28, 21, 18, 20, 14, 31, 27, 23, 15, 29, 17, 18, 27, 24, 26, 30, 20, 20, 20, 27, 28, 28, 26, 36, 19, 19, 20, 16, 11, 21, 33, 20, 33, 20, 35, 37, 15, 12, 34, 22, 25, 27, 35, 28, 18, 18, 32, 20, 27, 25, 34, 29, 13, 27, 28, 30, 14, 22, 30, 24, 23, 35, 25, 17, 37, 18, 26, 23, 31, 26, 35, 22, 17, 20, 31, 20, 4, 24, 17, 17, 13, 18, 21, 30, 15, 31, 22, 18, 27, 36, 19, 21, 13, 27, 24, 39, 32, 40, 35, 23, 39, 31, 26, 19, 20, 14, 23, 25, 20, 31, 37, 14, 38, 20, 21, 25, 12, 19, 30, 27, 18, 20, 19, 15, 35, 22, 38, 27, 35, 30, 19, 25, 28, 33, 23, 33, 35, 24, 16, 32, 19, 24, 29, 38, 18, 26, 18, 27, 29, 23, 30, 28, 24, 22, 20, 18, 25, 16, 26, 26, 20, 26, 31, 18, 28, 19, 28, 31, 19, 30, 25, 26, 38, 26, 38, 28, 30, 28, 27, 21, 20, 29, 20, 20, 22, 39, 3, 28, 26, 27, 17, 27, 23, 27, 23, 19, 25, 25, 19, 33, 25, 25, 28, 19, 32, 15, 27, 19, 20, 42, 21, 26, 14, 18, 25, 19, 25, 15, 30, 36, 28, 33, 20, 29, 41, 31, 32, 26, 20, 17, 27, 28, 28, 25, 18, 33, 12, 26, 26, 24, 19, 23, 20, 29, 32, 22, 34, 30, 13, 15, 28, 27, 19, 18, 15, 28, 21, 31, 24, 21, 23, 30, 30, 31, 18, 18, 19, 20, 28, 27, 40, 23, 31, 37, 27, 25, 17, 24, 17, 22, 18, 19, 21, 28, 22, 37, 31, 28, 26, 14, 33, 36, 21, 21, 21, 22, 9, 20, 29, 38, 28, 19, 28, 17, 17, 19, 20, 26, 27, 24, 19, 13, 20, 24, 16, 18, 24, 33, 18, 27, 24, 16, 28, 25, 26, 11, 15, 19, 28, 30, 14, 24, 29, 18, 27, 26, 13, 34, 32, 24, 30, 23, 24, 36, 21, 29, 21, 26, 15, 35, 31, 26, 13, 18, 20, 35, 31, 21, 39, 32, 22, 22, 27, 34, 19, 31, 25, 19, 21, 20, 27, 27, 29, 30, 18, 26, 12, 29, 27, 26, 25, 24, 33, 15, 31, 34, 20, 40, 21, 21, 40, 34, 29, 28, 19, 30, 22, 25, 27, 18, 34, 35, 25, 33, 27, 33, 15, 15, 35, 26, 15, 21, 28, 34, 21, 24, 16, 26, 19, 28, 25, 22, 16, 17, 20, 35, 21, 22, 24, 22, 30, 21, 21, 19, 25, 10, 27, 18, 18, 30, 19, 17, 23, 20, 25, 20, 33, 26, 16, 21, 17, 32, 22, 32, 28, 23, 34, 26, 25, 22, 18, 30, 21, 23, 24, 41, 17, 21, 44, 25, 21, 22, 29, 23, 31, 26, 29, 18, 31, 31, 17, 23, 30, 24, 36, 27, 24, 22, 30, 1, 14, 21, 13, 17, 22, 20, 24, 29, 27, 32, 29, 20, 20, 13, 22, 14, 26, 27, 28, 28, 37, 18, 28, 20, 30, 23, 19, 23, 15, 31, 31, 15, 37, 37, 18, 35, 22, 26, 36, 22, 26, 18, 22, 21, 28, 23, 14, 25, 21, 35, 30, 24, 15, 19, 19, 21, 34, 23, 26, 25, 22, 21, 19, 30, 22, 27, 24, 32, 22, 7, 13, 21, 34, 23, 36, 21, 21, 28, 17, 27, 18, 39, 19, 23, 26, 21, 14, 17, 31, 14, 19, 30, 17, 37, 16, 37, 39, 23, 25, 21, 30, 27, 27, 16, 20, 25, 18, 12, 29, 33, 17, 40, 16, 25, 30, 18, 23, 22, 29, 14, 23, 28, 17, 23, 26, 26, 34, 20, 25, 33, 25, 33, 26, 29, 24, 26, 23, 31, 33, 24, 13, 35, 20, 17, 16, 23, 19, 24, 25, 24, 20, 37, 26, 31, 30, 8, 20, 26, 29, 21, 22, 26, 37, 26, 31, 22, 17, 24, 18, 25, 33, 17, 27, 18, 30, 21, 24, 18, 22, 22, 28, 8, 19, 25, 13, 36, 20, 16, 28, 26, 29, 29, 24, 21, 21, 15, 35, 33, 16, 14, 22, 32, 36, 16, 15, 22, 33, 5, 25, 15, 20, 27, 23, 23, 14, 19, 20, 16, 25, 25, 26, 29, 33, 24, 26, 25, 30, 16, 21, 29, 12, 24, 20, 18, 25, 31, 25, 28, 23, 33, 29, 29, 44, 33, 19, 16, 19, 26, 31, 35, 32, 28, 24, 19, 33, 32, 31, 29, 21, 33, 22, 23, 19, 20, 34, 9, 19, 36, 15, 26, 27, 18, 22, 25, 12, 23, 39, 18, 17, 38, 19, 30, 31, 21, 22, 29, 18, 23, 24, 33, 34, 25, 33, 12, 15, 17, 34, 19, 0, 23, 37, 12, 24, 18, 28, 29, 14, 18, 19, 26, 23, 28, 20, 19, 27, 30, 33, 28, 23, 15, 38, 15, 23, 30, 5, 26, 37, 29, 23, 28, 25, 38, 18, 15, 22, 25, 28, 18, 40, 41, 38, 36, 25, 24, 19, 20, 23, 34, 25, 22, 24, 16, 21, 20, 18, 22, 25, 15, 38, 22, 36, 21, 23, 30, 19, 28, 29, 32, 0, 34, 28, 36, 23, 14, 12, 30, 18, 29, 19, 20, 21, 18, 18, 27, 34, 17, 28, 23, 26, 28, 30, 20, 36, 18, 27, 27, 15, 15, 32, 17, 20, 25, 29, 26, 24, 15, 29, 21, 16, 28, 27, 28, 40, 13, 15, 33, 14, 24, 37, 15, 27, 25, 24, 33, 18, 23, 31, 24, 28, 15, 38, 23, 21, 23, 30, 23, 25, 34, 21, 27, 34, 20, 23, 19, 21, 20, 26, 22, 19, 15, 33, 25, 16, 27, 21, 30, 26, 20, 28, 26, 22, 35, 18, 26, 26, 23, 34, 30, 30, 18, 22, 23, 17, 29, 23, 23, 29, 19, 13, 11, 7, 25, 15, 39, 19, 35, 20, 15, 13, 29, 26, 20, 13, 23, 34, 26, 25, 21, 26, 18, 19, 25, 19, 26, 28, 0, 4, 8, 3, 26, 35, 23, 24, 15, 20, 23, 30, 23, 34, 21, 20, 22, 26, 17, 29, 14, 22, 30, 28, 21, 32, 19, 22, 3, 30, 31, 19, 30, 19, 14, 19, 34, 31, 30, 28, 24, 29, 21, 26, 15, 36, 19, 29, 18, 21, 34, 26, 20, 25, 18, 25, 27, 2, 6, 30, 22, 31, 30, 26, 31, 17, 21, 31, 30, 14, 25, 25, 19, 27, 14, 15, 15, 19, 27, 14, 42, 19, 22, 40, 36, 15, 6, 35, 24, 20, 27, 29, 20, 24, 33, 17, 23, 26, 27, 26, 28, 31, 18, 20, 21, 18, 24, 30, 30, 28, 23, 26, 30, 18, 12, 19, 27, 25, 27, 15, 19, 17, 25, 21, 20, 19, 16, 25, 21, 26, 27, 14, 25, 29, 30, 46, 39, 27, 16, 37, 30, 2, 26, 19, 28, 34, 20, 24, 18, 19, 18, 21, 30, 19, 18, 29, 36, 25, 20, 18, 28, 20, 30, 24, 39, 18, 26, 37, 23, 23, 27, 24, 23, 27, 34, 2, 36, 26, 25, 25, 40, 22, 18, 22, 38, 23, 30, 26, 18, 24, 33, 30, 24, 3, 33, 22, 31, 22, 26, 21, 28, 28, 23, 22, 29, 24, 17, 24, 33, 30, 23, 35, 26, 19, 20, 23, 24, 27, 24, 26, 30, 22, 21, 22, 33, 18, 19, 25, 22, 12, 22, 14, 17, 10, 32, 22, 25, 21, 23, 16, 24, 33, 30, 34, 16, 18, 24, 26, 33, 33, 31, 18, 38, 24, 6, 33, 25, 29, 31, 21, 27, 23, 24, 20, 33, 37, 20, 15, 30, 24, 31, 35, 23, 32, 21, 22, 19, 20, 38, 35, 6, 18, 21, 19, 31, 23, 38, 17, 26, 23, 20, 25, 19, 29, 19, 21, 25, 35, 38, 24, 28, 24, 19, 30, 26, 25, 3, 34, 39, 9, 3, 27, 14, 35, 27, 30, 33, 23, 34, 22, 22, 24, 25, 25, 38, 31, 28, 30, 21, 24, 14, 21, 34, 25, 19, 31, 10, 20, 28, 17, 37, 13, 17, 25, 27, 22, 22, 21, 23, 30, 18, 19, 21, 29, 20, 16, 41, 39, 27, 33, 36, 25, 28, 5, 4, 24, 29, 23, 27, 18, 17, 30, 26, 22, 36, 28, 21, 18, 35, 19, 20, 13, 23, 32, 23, 26, 17, 30, 24, 39, 40, 3, 29, 35, 25, 21, 24, 25, 31, 24, 25, 29, 29, 13, 21, 19, 15, 24, 31, 20, 24, 20, 19, 34, 28, 23, 24, 39, 12, 35, 15, 19, 21, 18, 27, 31, 25, 21, 26, 21, 21, 24, 31, 23, 28, 26, 30, 16, 23, 19, 28, 17, 22, 13, 30, 15, 22, 32, 21, 26, 28, 23, 17, 20, 22, 20, 23, 28, 27, 20, 27, 22, 33, 39, 21, 30, 22, 39, 38, 24, 38, 35, 25, 32, 24, 21, 21, 29, 18, 37, 21, 21, 18, 15, 25, 21, 17, 26, 16, 26, 16, 17, 26, 17, 26, 36, 20, 23, 23, 30, 24, 32, 30, 24, 30, 30, 26, 31, 21, 15, 33, 18, 19, 24, 25, 23, 33, 24, 24, 28, 24, 23, 20, 28, 24, 27, 20, 18, 23, 20, 34, 28, 21, 28, 13, 24, 27, 36, 23, 22, 9, 2, 6, 13, 3]
      }]
  });

$('#4').highcharts({
      chart: {
          zoomType: 'x'
      },
      title: {
          text: ''
      },
      subtitle: {
          text: document.ontouchstart === undefined ?
                  'Click and drag in the plot area to zoom in' :
                  'Pinch the chart to zoom in'
      },
      xAxis: {
          type: 'datetime',
          minRange: 14 * 24 * 3600000 // fourteen days
      },
      yAxis: {
          title: {
              text: 'Happiness'
          }
      },
      legend: {
          enabled: false
      },
      plotOptions: {
          area: {
              fillColor: {
                  linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1},
                  stops: [
                      [0, Highcharts.getOptions().colors[0]],
                      [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
                  ]
              },
              marker: {
                  radius: 2
              },
              lineWidth: 1,
              states: {
                  hover: {
                      lineWidth: 1
                  }
              },
              threshold: null
          }
      },

      series: [{
          type: 'area',
          name: '',
          pointInterval: 24 * 3600 * 1000,
          pointStart: Date.UTC(2010, 0, 0),
          data: [13, 8, 16, 23, 13, 8, 19, 1, 10, 6, 5, 7, 27, 11, 5, 10, 26, 16, 10, 0, 10, 16, 0, 15, 8, 9, 2, 11, 23, 22, 15, 12, 24, 8, 18, 2, 21, 26, 20, 14, 11, 4, 26, 11, 16, 13, 7, 11, 12, 22, 3, 1, 11, 4, 12, 6, 16, 19, 33, 0, 30, 7, 15, 33, 4, 29, 11, 6, 23, 27, 20, 4, 13, 23, 5, 31, 5, 10, 15, 6, 9, 26, 22, 4, 18, 16, 2, 17, 21, 13, 28, 8, 9, 21, 5, 17, 35, 22, 18, 2, 17, 24, 17, 20, 4, 12, 3, 14, 30, 5, 12, 10, 20, 28, 5, 11, 7, 7, 8, 25, 14, 20, 2, 6, 25, 24, 21, 1, 3, 12, 7, 26, 27, 19, 6, 8, 6, 14, 23, 9, 23, 21, 25, 2, 21, 6, 25, 18, 4, 5, 15, 27, 15, 3, 3, 1, 12, 7, 29, 23, 1, 10, 10, 10, 1, 4, 27, 17, 1, 1, 22, 6, 15, 2, 11, 13, 8, 16, 15, 27, 3, 10, 25, 18, 24, 4, 19, 0, 15, 19, 25, 4, 14, 14, 32, 2, 16, 8, 5, 5, 19, 0, 1, 35, 13, 21, 10, 3, 5, 3, 17, 2, 21, 0, 1, 17, 26, 14, 4, 18, 26, 9, 15, 14, 0, 25, 3, 21, 28, 6, 15, 11, 7, 2, 24, 14, 22, 7, 6, 14, 11, 3, 10, 7, 9, 5, 9, 8, 19, 23, 23, 22, 8, 11, 24, 7, 16, 8, 34, 9, 21, 9, 16, 4, 4, 7, 17, 32, 20, 9, 22, 10, 19, 6, 0, 12, 8, 27, 10, 18, 12, 10, 28, 9, 0, 21, 5, 6, 7, 11, 16, 4, 15, 33, 11, 19, 16, 37, 5, 27, 24, 20, 24, 9, 2, 9, 24, 26, 14, 18, 21, 14, 6, 28, 12, 26, 4, 9, 9, 27, 17, 17, 25, 0, 6, 8, 6, 22, 13, 5, 29, 18, 2, 29, 13, 11, 3, 29, 7, 0, 1, 7, 10, 10, 1, 5, 8, 3, 17, 21, 4, 4, 7, 17, 1, 4, 27, 30, 16, 7, 1, 10, 30, 28, 1, 28, 29, 3, 11, 1, 17, 23, 19, 1, 18, 2, 24, 12, 14, 7, 7, 11, 5, 15, 15, 31, 14, 5, 5, 1, 8, 22, 9, 32, 9, 14, 7, 14, 21, 12, 16, 25, 28, 15, 3, 0, 33, 19, 5, 9, 10, 24, 9, 15, 18, 24, 23, 18, 15, 17, 19, 22, 38, 23, 1, 19, 3, 5, 3, 16, 6, 20, 19, 20, 5, 9, 26, 8, 14, 16, 6, 15, 7, 12, 7, 14, 6, 2, 17, 5, 1, 33, 26, 18, 4, 4, 15, 9, 6, 28, 5, 15, 10, 24, 4, 26, 9, 14, 34, 1, 4, 21, 20, 10, 20, 6, 21, 27, 25, 8, 4, 12, 3, 3, 28, 4, 16, 15, 7, 7, 8, 3, 1, 4, 24, 11, 28, 31, 7, 6, 20, 13, 5, 24, 3, 15, 14, 7, 10, 26, 0, 13, 4, 25, 6, 7, 32, 0, 2, 24, 10, 6, 3, 26, 0, 2, 7, 0, 26, 2, 31, 5, 6, 4, 2, 5, 10, 16, 14, 19, 25, 19, 1, 20, 10, 18, 4, 18, 17, 6, 11, 30, 34, 57, 36, 51, 52, 88, 16, 6, 9, 3, 28, 12, 55, 71, 40, 59, 50, 58, 58, 49, 41, 41, 37, 45, 84, 62, 55, 58, 59, 61, 28, 36, 23, 23, 47, 96, 4, 16, 6, 2, 21, 6, 3, 29, 8, 12, 4, 18, 25, 8, 28, 23, 6, 28, 2, 10, 13, 20, 9, 13, 6, 7, 12, 11, 17, 8, 20, 32, 19, 21, 47, 52, 6, 5, 9, 18, 16, 4, 6, 21, 26, 2, 11, 3, 0, 3, 13, 8, 6, 14, 5, 30, 20, 27, 10, 14, 12, 27, 9, 10, 6, 12, 17, 6, 16, 8, 11, 16, 0, 16, 12, 8, 17, 15, 15, 1, 15, 25, 1, 5, 10, 19, 9, 13, 12, 15, 7, 22, 13, 22, 26, 17, 7, 2, 14, 9, 10, 26, 8, 2, 9, 21, 6, 19, 1, 9, 1, 26, 5, 19, 27, 3, 13, 14, 5, 8, 0, 32, 22, 0, 32, 2, 10, 0, 1, 3, 16, 27, 9, 20, 7, 2, 5, 22, 7, 2, 22, 27, 19, 5, 13, 23, 6, 6, 3, 11, 32, 8, 20, 21, 19, 15, 23, 1, 13, 5, 24, 13, 13, 15, 4, 15, 23, 16, 4, 3, 9, 4, 16, 2, 5, 3, 25, 6, 9, 17, 4, 6, 8, 20, 6, 1, 18, 34, 2, 4, 22, 28, 29, 21, 7, 9, 9, 1, 21, 1, 12, 15, 11, 12, 4, 2, 23, 6, 4, 2, 11, 14, 17, 15, 11, 11, 12, 21, 4, 28, 8, 23, 20, 28, 6, 24, 1, 17, 1, 18, 12, 2, 14, 4, 23, 4, 32, 7, 22, 24, 8, 17, 34, 10, 14, 21, 22, 26, 13, 2, 27, 31, 6, 23, 6, 16, 0, 25, 14, 20, 33, 10, 6, 14, 13, 15, 8, 1, 17, 26, 15, 9, 3, 14, 10, 28, 5, 2, 25, 8, 1, 1, 10, 22, 23, 24, 11, 19, 11, 0, 4, 27, 11, 3, 18, 9, 8, 1, 14, 20, 8, 12, 30, 23, 12, 19, 7, 10, 11, 4, 19, 2, 2, 13, 10, 1, 4, 13, 21, 3, 17, 7, 9, 9, 12, 16, 21, 12, 4, 7, 12, 19, 6, 2, 8, 5, 15, 13, 7, 17, 7, 11, 25, 31, 12, 17, 21, 0, 1, 18, 0, 11, 12, 26, 13, 5, 27, 15, 1, 14, 6, 5, 22, 9, 0, 5, 5, 15, 5, 2, 3, 1, 2, 0, 11, 4, 6, 0, 18, 18, 23, 10, 4, 15, 11, 12, 40, 2, 14, 8, 1, 23, 0, 27, 11, 2, 20, 26, 11, 1, 2, 15, 17, 21, 7, 19, 25, 6, 29, 11, 5, 3, 5, 14, 14, 18, 15, 13, 7, 35, 7, 5, 13, 22, 19, 17, 4, 29, 7, 4, 9, 23, 4, 6, 20, 4, 1, 12, 9, 11, 3, 25, 18, 20, 14, 28, 24, 1, 25, 14, 25, 1, 20, 2, 23, 12, 19, 13, 10, 12, 13, 3, 27, 9, 18, 19, 27, 9, 16, 1, 6, 4, 30, 12, 17, 19, 3, 5, 2, 11, 16, 11, 25, 19, 22, 15, 21, 5, 26, 23, 17, 28, 9, 26, 15, 15, 3, 23, 23, 8, 18, 3, 15, 24, 1, 6, 9, 4, 6, 24, 13, 33, 6, 15, 5, 23, 6, 7, 11, 14, 8, 2, 9, 18, 7, 19, 3, 20, 4, 15, 13, 10, 21, 2, 8, 25, 15, 7, 27, 25, 4, 16, 3, 25, 26, 1, 35, 17, 22, 19, 17, 6, 6, 4, 20, 1, 8, 22, 4, 9, 1, 2, 7, 25, 24, 0, 10, 14, 4, 33, 9, 8, 20, 15, 10, 20, 7, 1, 12, 2, 21, 8, 10, 10, 9, 18, 11, 18, 15, 15, 9, 8, 28, 7, 11, 5, 1, 8, 16, 8, 21, 4, 9, 25, 18, 19, 11, 25, 1, 10, 29, 22, 7, 7, 6, 9, 18, 17, 15, 11, 8, 8, 9, 19, 9, 4, 4, 17, 7, 9, 5, 42, 4, 22, 11, 14, 8, 22, 11, 7, 5, 35, 22, 13, 50, 21, 17, 19, 20, 15, 21, 8, 4, 27, 1, 1, 29, 0, 25, 15, 20, 19, 12, 23, 10, 27, 9, 2, 13, 9, 39, 30, 20, 43, 43, 7, 19, 17, 16, 12, 31, 5, 3, 21, 1, 6, 15, 9, 13, 22, 4, 15, 17, 9, 7, 3, 7, 13, 6, 26, 5, 21, 7, 4, 5, 1, 14, 2, 25, 18, 28, 16, 25, 35, 16, 22, 10, 15, 1, 7, 29, 19, 11, 23, 17, 10, 14, 13, 16, 4, 13, 12, 25, 11, 7, 19, 15, 7, 17, 1, 5, 24, 20, 2, 7, 12, 28, 32, 17, 25, 1, 27, 15, 35, 8, 3, 13, 12, 20, 6, 3, 19, 3, 9, 24, 2, 6, 22, 14, 72, 25, 23, 25, 4, 23, 16, 11, 2, 21, 29, 22, 24, 19, 17, 2, 11, 5, 13, 12, 1, 23, 14, 11, 8, 21, 5, 30, 23, 8, 0, 2, 8, 3, 8, 17, 17, 16, 10, 25, 7, 34, 25, 19, 4, 27, 11, 2, 4, 28, 31, 6, 9, 30, 11, 4, 1, 27, 14, 12, 14, 8, 26, 13, 12, 4, 24, 12, 21, 27, 10, 4, 3, 15, 30, 5, 31, 19, 28, 17, 27, 13, 17, 21, 4, 22, 17, 4, 20, 2, 23, 17, 20, 23, 3, 20, 3, 11, 29, 1, 14, 14, 36, 12, 24, 16, 0, 7, 20, 20, 8, 9, 14, 5, 3, 16, 9, 26, 7, 13, 19, 11, 16, 33, 7, 1, 17, 0, 17, 11, 13, 24, 2, 23, 3, 29, 5, 26, 9, 5, 24, 6, 18, 11, 3, 7, 12, 5, 17, 24, 2, 7, 18, 15, 10, 12, 8, 26, 16, 11, 15, 3, 10, 14, 3, 8, 9, 15, 4, 15, 8, 8, 15, 24, 0, 8, 5, 20, 4, 26, 30, 14, 7, 1, 5, 14, 21, 9, 25, 11, 11, 20, 27, 12, 35, 16, 25, 20, 17, 18, 2, 19, 17, 13, 19, 12, 16, 25, 23, 1, 25, 17, 6, 3, 30, 16, 12, 5, 16, 9, 29, 26, 25, 23, 19, 5, 25, 19, 3, 16, 16, 13, 21, 19, 29, 2, 28, 5, 5, 8, 25, 10, 9, 1, 17, 26, 19, 14, 19, 25, 9, 14, 21, 25, 9, 28, 9, 26, 32, 22, 14, 2, 16, 19, 1, 24, 8, 18, 23, 24, 12, 8, 2, 8, 16, 22, 27, 21, 11, 8, 15, 19, 25, 24, 16, 30, 0, 13, 4, 7, 11, 4, 13, 23, 23, 21, 28, 12, 25, 19, 18, 7, 14, 6, 3, 3, 23, 17, 25, 1, 22, 18, 7, 25, 32, 14, 8, 13, 2, 5, 7, 26, 9, 1, 21, 27, 12, 30, 17, 30, 19, 3, 10, 33, 20, 4, 6, 20, 21, 2, 20, 28, 13, 16, 12, 9, 10, 34, 12, 22, 23, 7, 5, 22, 30, 2, 28, 19, 7, 22, 15, 2, 0, 17, 25, 23, 8, 7, 14, 1, 32, 8, 21, 3, 13, 12, 19, 17, 12, 1, 1, 16, 1, 25, 15, 3, 2, 24, 25, 11, 25, 5, 18, 4, 23, 3, 5, 1, 12, 2, 21, 12, 29, 14, 9, 20, 27, 14, 8, 16, 10, 16, 26, 14, 13, 32, 20, 6, 6, 31, 11, 33, 19, 23, 13, 10, 20, 7, 24, 2, 3, 15, 2, 7, 9, 15, 4, 7, 8, 2, 20, 9, 10, 10, 14, 3, 2, 13, 11, 22, 10, 9, 21, 13, 21, 18, 4, 27, 12, 11, 8, 20, 27, 15, 18, 1, 3, 13, 32, 48]
      }]
  });

$('#5').highcharts({
      chart: {
          zoomType: 'x'
      },
      title: {
          text: ''
      },
      subtitle: {
          text: document.ontouchstart === undefined ?
                  'Click and drag in the plot area to zoom in' :
                  'Pinch the chart to zoom in'
      },
      xAxis: {
          type: 'datetime',
          minRange: 14 * 24 * 3600000 // fourteen days
      },
      yAxis: {
          title: {
              text: 'Happiness'
          }
      },
      legend: {
          enabled: false
      },
      plotOptions: {
          area: {
              fillColor: {
                  linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1},
                  stops: [
                      [0, Highcharts.getOptions().colors[0]],
                      [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
                  ]
              },
              marker: {
                  radius: 2
              },
              lineWidth: 1,
              states: {
                  hover: {
                      lineWidth: 1
                  }
              },
              threshold: null
          }
      },

      series: [{
          type: 'area',
          name: '',
          pointInterval: 24 * 3600 * 1000,
          pointStart: Date.UTC(2010, 0, 0),
          data: [16, 9, 24, 23, 20, 15, 20, 8, 13, 9, 5, 7, 26, 14, 8, 18, 25, 19, 14, 0, 13, 24, 1, 17, 9, 16, 5, 11, 24, 24, 19, 18, 24, 13, 19, 7, 21, 26, 19, 21, 19, 12, 27, 16, 22, 14, 15, 18, 19, 20, 4, 9, 13, 12, 20, 11, 19, 18, 31, 1, 30, 14, 17, 32, 4, 30, 17, 9, 25, 25, 21, 7, 16, 22, 10, 30, 8, 14, 23, 7, 16, 24, 20, 5, 18, 20, 7, 16, 22, 14, 27, 8, 11, 20, 6, 15, 34, 20, 19, 4, 19, 22, 15, 18, 7, 19, 6, 19, 31, 11, 14, 12, 19, 30, 7, 13, 9, 7, 13, 24, 15, 21, 4, 9, 23, 25, 23, 7, 7, 16, 7, 28, 27, 17, 8, 16, 12, 15, 25, 17, 22, 19, 27, 7, 21, 7, 27, 17, 7, 12, 23, 25, 15, 11, 7, 3, 16, 7, 30, 24, 8, 11, 14, 11, 7, 5, 25, 18, 4, 4, 22, 9, 21, 5, 17, 13, 12, 23, 19, 28, 5, 10, 24, 17, 26, 6, 21, 4, 19, 18, 25, 9, 17, 21, 32, 4, 22, 12, 5, 7, 18, 6, 6, 37, 14, 20, 11, 5, 13, 9, 18, 7, 23, 4, 6, 17, 24, 18, 6, 19, 28, 16, 23, 22, 3, 23, 4, 23, 30, 13, 17, 13, 15, 8, 26, 21, 23, 14, 11, 20, 12, 3, 17, 9, 12, 8, 13, 12, 20, 22, 24, 20, 10, 11, 25, 9, 18, 11, 33, 13, 23, 9, 17, 9, 7, 11, 18, 34, 18, 17, 21, 14, 17, 12, 8, 17, 13, 28, 13, 16, 17, 11, 26, 12, 4, 20, 6, 13, 10, 14, 20, 11, 18, 32, 17, 18, 23, 39, 11, 26, 26, 21, 23, 15, 8, 17, 26, 24, 16, 20, 20, 19, 7, 26, 16, 24, 4, 12, 16, 29, 18, 19, 23, 0, 6, 11, 7, 20, 17, 10, 28, 20, 2, 31, 14, 19, 11, 30, 10, 2, 6, 13, 10, 18, 8, 7, 14, 7, 15, 19, 4, 11, 15, 17, 4, 10, 29, 32, 23, 15, 2, 17, 28, 27, 7, 29, 27, 10, 19, 4, 19, 22, 19, 2, 16, 8, 25, 14, 17, 11, 12, 13, 7, 17, 23, 30, 14, 7, 6, 5, 10, 23, 9, 32, 12, 20, 14, 14, 19, 16, 19, 26, 30, 20, 11, 0, 32, 18, 10, 12, 10, 26, 15, 20, 20, 25, 22, 19, 18, 17, 18, 24, 38, 23, 7, 20, 11, 5, 9, 23, 8, 21, 20, 21, 12, 15, 24, 16, 20, 17, 12, 15, 15, 15, 15, 17, 6, 9, 16, 11, 9, 32, 24, 17, 11, 8, 21, 16, 7, 26, 6, 15, 17, 23, 7, 26, 10, 19, 34, 2, 6, 22, 22, 18, 19, 12, 19, 26, 27, 13, 6, 12, 3, 8, 30, 8, 18, 19, 8, 12, 16, 7, 8, 12, 26, 14, 30, 30, 10, 9, 19, 18, 9, 25, 8, 17, 20, 11, 13, 24, 7, 13, 11, 27, 11, 12, 34, 6, 9, 23, 11, 14, 8, 24, 4, 8, 7, 5, 24, 5, 33, 9, 14, 5, 6, 12, 14, 22, 16, 17, 26, 20, 9, 22, 12, 20, 12, 18, 18, 10, 12, 30, 35, 57, 36, 53, 51, 88, 19, 12, 13, 10, 28, 12, 56, 71, 40, 61, 50, 57, 58, 49, 42, 39, 38, 43, 84, 60, 55, 59, 57, 62, 30, 34, 24, 22, 48, 95, 8, 22, 14, 7, 22, 13, 4, 30, 10, 15, 5, 16, 27, 10, 29, 21, 14, 26, 3, 13, 18, 20, 17, 18, 9, 11, 20, 16, 17, 11, 19, 34, 19, 21, 48, 52, 7, 13, 11, 18, 19, 8, 10, 20, 26, 7, 13, 4, 2, 3, 13, 9, 11, 16, 8, 32, 19, 28, 11, 22, 15, 27, 17, 13, 14, 14, 19, 6, 19, 15, 11, 16, 0, 23, 12, 11, 16, 15, 18, 8, 16, 25, 6, 5, 14, 20, 14, 14, 20, 21, 8, 21, 16, 21, 27, 19, 7, 3, 16, 10, 15, 26, 13, 10, 11, 21, 12, 18, 2, 9, 9, 24, 11, 17, 28, 5, 21, 16, 7, 9, 7, 34, 24, 1, 32, 6, 10, 2, 7, 5, 16, 28, 16, 18, 8, 7, 5, 23, 8, 10, 23, 26, 17, 10, 18, 23, 12, 10, 5, 11, 33, 14, 21, 20, 19, 18, 23, 5, 18, 7, 23, 16, 15, 18, 10, 15, 21, 18, 5, 9, 11, 5, 19, 6, 6, 10, 27, 9, 14, 17, 9, 7, 11, 18, 9, 3, 19, 34, 9, 10, 22, 27, 30, 20, 8, 9, 14, 9, 21, 3, 18, 23, 17, 17, 8, 3, 24, 9, 9, 6, 15, 19, 15, 20, 12, 13, 20, 23, 5, 30, 14, 21, 18, 28, 6, 25, 8, 17, 2, 16, 18, 5, 20, 4, 22, 6, 34, 14, 22, 23, 8, 17, 36, 15, 17, 21, 24, 27, 13, 10, 27, 31, 11, 22, 12, 24, 4, 23, 20, 19, 35, 16, 14, 21, 17, 21, 16, 2, 16, 27, 20, 14, 11, 20, 11, 27, 10, 5, 27, 11, 2, 1, 17, 20, 21, 24, 16, 20, 12, 6, 9, 28, 16, 7, 19, 17, 12, 3, 15, 20, 9, 17, 31, 21, 17, 17, 14, 12, 12, 9, 17, 9, 5, 16, 16, 7, 9, 16, 21, 7, 17, 12, 9, 11, 13, 20, 20, 19, 11, 10, 16, 19, 13, 5, 14, 6, 17, 14, 10, 17, 7, 19, 26, 31, 12, 18, 22, 5, 3, 18, 0, 13, 18, 27, 15, 12, 28, 23, 1, 19, 7, 5, 23, 17, 1, 8, 8, 18, 5, 10, 3, 9, 3, 4, 18, 11, 10, 2, 20, 16, 24, 10, 10, 18, 18, 17, 40, 6, 15, 8, 5, 24, 7, 28, 11, 2, 21, 25, 11, 2, 2, 16, 17, 22, 9, 21, 23, 14, 27, 12, 5, 3, 7, 15, 17, 17, 20, 18, 7, 36, 12, 7, 16, 21, 18, 18, 6, 29, 13, 7, 13, 25, 7, 8, 21, 11, 3, 12, 9, 16, 4, 24, 17, 20, 17, 26, 23, 8, 23, 17, 26, 9, 18, 10, 24, 19, 20, 16, 13, 17, 15, 6, 27, 12, 20, 20, 25, 17, 20, 9, 9, 7, 29, 15, 18, 17, 7, 13, 5, 11, 16, 14, 27, 21, 24, 18, 23, 13, 25, 23, 16, 26, 16, 28, 15, 17, 10, 24, 24, 11, 16, 3, 23, 25, 2, 10, 11, 6, 11, 23, 20, 35, 7, 19, 7, 22, 6, 10, 15, 14, 13, 3, 9, 18, 11, 18, 7, 18, 11, 23, 19, 17, 22, 3, 16, 26, 17, 15, 25, 23, 11, 17, 11, 23, 25, 2, 34, 19, 24, 20, 18, 9, 6, 11, 18, 3, 13, 20, 5, 15, 6, 10, 12, 25, 24, 2, 17, 22, 9, 31, 13, 9, 21, 17, 12, 22, 8, 4, 15, 3, 20, 15, 15, 17, 12, 17, 16, 19, 17, 20, 12, 10, 30, 8, 14, 9, 8, 10, 21, 13, 21, 7, 10, 25, 17, 19, 12, 23, 7, 18, 31, 20, 9, 14, 8, 17, 18, 17, 23, 17, 16, 13, 15, 18, 17, 10, 7, 18, 13, 9, 12, 44, 6, 20, 16, 19, 15, 21, 13, 8, 5, 35, 20, 16, 49, 23, 19, 20, 19, 19, 19, 14, 8, 26, 4, 9, 29, 6, 23, 19, 18, 17, 20, 23, 18, 25, 13, 9, 14, 9, 37, 30, 19, 44, 44, 7, 20, 15, 16, 15, 31, 9, 11, 19, 4, 8, 16, 9, 19, 22, 6, 17, 19, 11, 9, 4, 13, 19, 12, 25, 9, 23, 13, 5, 13, 1, 14, 10, 27, 17, 30, 18, 25, 35, 23, 20, 18, 23, 4, 12, 31, 20, 17, 24, 18, 16, 19, 21, 21, 4, 20, 20, 26, 18, 12, 18, 20, 15, 15, 3, 7, 24, 19, 3, 9, 15, 28, 30, 16, 26, 5, 25, 21, 37, 8, 6, 16, 17, 18, 14, 8, 17, 8, 16, 22, 8, 10, 21, 16, 74, 23, 22, 26, 4, 22, 17, 16, 3, 19, 29, 24, 24, 19, 19, 4, 17, 13, 15, 12, 9, 21, 16, 17, 10, 22, 12, 29, 22, 8, 1, 9, 10, 8, 14, 17, 15, 21, 18, 24, 7, 36, 27, 19, 7, 28, 14, 7, 5, 27, 32, 8, 17, 32, 11, 10, 6, 26, 16, 15, 19, 14, 28, 20, 17, 11, 25, 20, 22, 28, 13, 11, 4, 23, 29, 7, 31, 17, 30, 15, 28, 15, 19, 23, 9, 24, 16, 7, 20, 9, 22, 16, 18, 24, 11, 20, 6, 15, 30, 1, 20, 20, 38, 13, 23, 22, 4, 8, 21, 22, 13, 10, 15, 10, 11, 24, 9, 27, 12, 15, 21, 16, 16, 31, 13, 6, 18, 3, 17, 14, 19, 24, 2, 24, 3, 29, 8, 27, 9, 13, 24, 7, 19, 11, 10, 10, 20, 12, 18, 25, 4, 7, 19, 22, 10, 20, 9, 26, 24, 12, 20, 3, 17, 22, 10, 13, 13, 15, 6, 15, 13, 8, 19, 24, 5, 11, 12, 20, 6, 27, 31, 21, 14, 8, 6, 17, 23, 11, 23, 15, 13, 18, 28, 15, 34, 22, 24, 19, 15, 20, 7, 19, 15, 21, 17, 15, 19, 23, 24, 4, 25, 17, 13, 3, 32, 16, 14, 12, 17, 17, 29, 28, 27, 22, 17, 6, 27, 19, 5, 22, 18, 17, 21, 18, 28, 9, 28, 7, 7, 16, 25, 10, 17, 8, 17, 28, 17, 17, 18, 25, 17, 19, 20, 26, 14, 28, 15, 28, 34, 20, 16, 3, 21, 20, 7, 25, 12, 17, 25, 22, 13, 13, 7, 11, 22, 20, 25, 22, 12, 12, 18, 18, 24, 24, 24, 29, 0, 16, 5, 13, 17, 9, 15, 22, 25, 19, 29, 18, 25, 18, 19, 13, 19, 8, 3, 4, 22, 18, 24, 7, 21, 18, 13, 27, 34, 22, 8, 14, 2, 12, 14, 24, 11, 7, 19, 25, 17, 32, 17, 31, 18, 10, 15, 35, 20, 5, 12, 21, 22, 7, 18, 26, 18, 19, 14, 11, 10, 33, 18, 22, 21, 9, 9, 22, 31, 8, 27, 17, 13, 22, 16, 8, 1, 15, 25, 24, 11, 7, 17, 8, 33, 14, 23, 10, 16, 17, 20, 15, 20, 5, 2, 23, 9, 24, 22, 7, 9, 25, 25, 19, 23, 11, 20, 5, 22, 7, 12, 7, 12, 2, 19, 13, 31, 22, 14, 19, 25, 16, 13, 21, 13, 24, 25, 21, 16, 32, 19, 8, 8, 32, 16, 34, 19, 21, 14, 15, 21, 7, 22, 3, 4, 18, 7, 13, 15, 19, 8, 11, 16, 6, 20, 16, 10, 10, 18, 8, 4, 16, 15, 23, 15, 13, 20, 14, 21, 16, 11, 29, 17, 17, 8, 22, 27, 16, 17, 7, 11, 19, 32, 46]
      }]
  });

$('#6').highcharts({
      chart: {
          zoomType: 'x'
      },
      title: {
          text: ''
      },
      subtitle: {
          text: document.ontouchstart === undefined ?
                  'Click and drag in the plot area to zoom in' :
                  'Pinch the chart to zoom in'
      },
      xAxis: {
          type: 'datetime',
          minRange: 14 * 24 * 3600000 // fourteen days
      },
      yAxis: {
          title: {
              text: 'Happiness'
          }
      },
      legend: {
          enabled: false
      },
      plotOptions: {
          area: {
              fillColor: {
                  linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1},
                  stops: [
                      [0, Highcharts.getOptions().colors[0]],
                      [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
                  ]
              },
              marker: {
                  radius: 2
              },
              lineWidth: 1,
              states: {
                  hover: {
                      lineWidth: 1
                  }
              },
              threshold: null
          }
      },

      series: [{
          type: 'area',
          name: '',
          pointInterval: 24 * 3600 * 1000,
          pointStart: Date.UTC(2010, 0, 0),
          data: [18, 11, 23, 23, 20, 22, 21, 8, 16, 16, 5, 11, 24, 14, 8, 19, 25, 20, 15, 6, 20, 24, 9, 16, 9, 18, 5, 19, 26, 22, 17, 18, 22, 18, 21, 13, 22, 24, 20, 22, 18, 20, 25, 24, 22, 22, 15, 17, 18, 21, 11, 17, 14, 14, 18, 18, 20, 18, 32, 4, 28, 16, 17, 31, 9, 31, 18, 16, 25, 24, 21, 10, 24, 23, 14, 30, 8, 20, 25, 8, 17, 22, 22, 13, 17, 21, 15, 17, 20, 16, 29, 12, 16, 21, 11, 21, 32, 21, 21, 7, 21, 24, 21, 17, 8, 20, 14, 17, 32, 18, 20, 15, 18, 29, 11, 15, 11, 15, 19, 24, 21, 23, 11, 9, 24, 26, 23, 8, 8, 17, 14, 29, 29, 15, 14, 20, 18, 18, 24, 16, 21, 19, 29, 15, 21, 8, 25, 16, 15, 18, 25, 26, 15, 15, 15, 8, 22, 10, 30, 24, 9, 13, 18, 17, 14, 6, 25, 16, 9, 4, 20, 15, 23, 11, 18, 15, 12, 21, 19, 30, 8, 12, 22, 18, 25, 12, 21, 10, 18, 17, 24, 12, 19, 22, 33, 12, 23, 16, 13, 7, 17, 14, 11, 38, 22, 18, 14, 7, 15, 16, 19, 13, 22, 11, 13, 18, 25, 18, 11, 21, 26, 18, 23, 20, 10, 24, 10, 25, 31, 17, 17, 20, 19, 10, 27, 22, 23, 17, 19, 22, 13, 9, 19, 10, 13, 13, 21, 12, 20, 23, 25, 22, 13, 11, 26, 13, 17, 15, 35, 13, 24, 14, 16, 17, 12, 13, 20, 34, 20, 18, 22, 15, 17, 20, 14, 18, 19, 26, 16, 24, 19, 13, 25, 15, 11, 19, 13, 19, 16, 20, 21, 16, 16, 30, 17, 19, 23, 37, 13, 24, 25, 22, 21, 21, 14, 15, 25, 24, 21, 22, 21, 21, 15, 25, 24, 22, 10, 19, 23, 31, 18, 21, 25, 1, 10, 17, 8, 21, 16, 18, 27, 19, 7, 29, 17, 20, 14, 32, 13, 2, 13, 14, 14, 20, 13, 8, 18, 7, 21, 21, 6, 12, 17, 16, 7, 17, 29, 30, 24, 20, 8, 18, 29, 26, 8, 29, 29, 12, 19, 9, 18, 22, 21, 10, 16, 8, 25, 16, 16, 13, 14, 18, 7, 18, 25, 31, 14, 12, 13, 5, 10, 21, 12, 33, 15, 20, 21, 18, 19, 16, 21, 27, 28, 19, 17, 3, 32, 18, 17, 17, 18, 25, 23, 19, 18, 24, 21, 17, 19, 15, 16, 25, 36, 21, 9, 18, 17, 12, 11, 23, 11, 19, 18, 21, 18, 18, 26, 20, 20, 19, 17, 23, 19, 23, 15, 16, 8, 17, 16, 11, 13, 32, 25, 17, 16, 16, 20, 17, 7, 25, 6, 23, 17, 21, 12, 28, 13, 18, 35, 7, 7, 21, 21, 18, 17, 14, 18, 27, 25, 17, 13, 12, 8, 12, 28, 12, 16, 17, 10, 14, 17, 12, 10, 16, 25, 16, 30, 30, 10, 13, 18, 16, 17, 26, 12, 17, 19, 13, 21, 22, 14, 17, 14, 26, 11, 14, 35, 12, 17, 22, 15, 16, 11, 23, 11, 15, 14, 6, 22, 7, 35, 17, 19, 7, 8, 14, 20, 20, 21, 15, 24, 19, 17, 23, 13, 18, 18, 18, 16, 11, 14, 31, 34, 55, 38, 51, 51, 86, 18, 12, 21, 18, 28, 20, 55, 72, 41, 59, 49, 55, 60, 48, 44, 40, 36, 45, 83, 62, 57, 60, 58, 62, 28, 34, 23, 21, 49, 94, 14, 23, 14, 7, 23, 14, 5, 31, 10, 18, 8, 18, 25, 11, 28, 19, 22, 24, 4, 17, 16, 20, 15, 17, 15, 12, 20, 20, 18, 19, 21, 36, 21, 19, 46, 50, 15, 14, 18, 20, 19, 9, 15, 18, 25, 15, 16, 11, 2, 7, 18, 10, 12, 24, 13, 31, 19, 26, 11, 21, 16, 29, 15, 19, 21, 17, 18, 7, 20, 22, 14, 23, 5, 21, 19, 12, 21, 20, 17, 8, 20, 24, 10, 8, 14, 22, 22, 15, 22, 19, 10, 22, 19, 22, 27, 18, 8, 9, 18, 15, 16, 24, 21, 17, 14, 20, 18, 19, 3, 11, 10, 22, 16, 16, 27, 11, 22, 20, 11, 13, 9, 36, 24, 2, 30, 10, 16, 6, 8, 11, 21, 30, 24, 17, 8, 8, 6, 21, 14, 13, 21, 24, 15, 17, 19, 24, 14, 14, 12, 15, 32, 15, 21, 22, 17, 19, 22, 8, 16, 8, 23, 19, 20, 20, 12, 15, 22, 18, 5, 10, 14, 12, 19, 8, 14, 16, 26, 9, 14, 17, 14, 13, 14, 20, 9, 10, 18, 32, 17, 11, 24, 25, 31, 22, 11, 16, 22, 15, 23, 5, 16, 23, 19, 16, 16, 3, 23, 16, 12, 9, 20, 17, 18, 21, 20, 14, 21, 25, 12, 31, 18, 19, 17, 29, 12, 25, 12, 15, 3, 18, 19, 8, 21, 10, 23, 10, 32, 14, 24, 25, 13, 15, 37, 17, 18, 22, 25, 25, 19, 15, 29, 31, 11, 22, 15, 22, 8, 22, 18, 18, 33, 21, 21, 19, 17, 23, 19, 2, 20, 26, 19, 18, 11, 21, 11, 27, 11, 8, 25, 11, 6, 7, 15, 18, 21, 24, 20, 18, 16, 10, 9, 28, 23, 8, 17, 19, 12, 10, 18, 20, 10, 19, 32, 21, 15, 15, 21, 12, 15, 14, 16, 11, 13, 21, 17, 13, 12, 17, 22, 7, 19, 17, 16, 18, 18, 21, 20, 21, 11, 11, 21, 21, 13, 8, 14, 10, 18, 17, 15, 15, 10, 19, 26, 32, 14, 16, 22, 7, 3, 20, 1, 16, 20, 25, 18, 12, 26, 22, 9, 19, 14, 9, 24, 16, 8, 13, 14, 18, 7, 13, 5, 15, 3, 7, 19, 18, 11, 6, 19, 22, 25, 15, 16, 20, 20, 16, 39, 12, 20, 12, 5, 24, 9, 29, 14, 9, 20, 24, 18, 10, 10, 24, 16, 24, 17, 21, 25, 16, 28, 13, 10, 9, 13, 23, 16, 17, 21, 18, 11, 35, 17, 8, 24, 22, 16, 18, 12, 27, 21, 9, 15, 27, 14, 15, 20, 11, 3, 19, 16, 18, 4, 24, 15, 18, 18, 28, 24, 9, 25, 17, 26, 14, 16, 18, 25, 18, 19, 17, 19, 17, 18, 13, 25, 13, 20, 18, 23, 15, 18, 9, 12, 15, 27, 19, 20, 18, 13, 16, 10, 18, 17, 17, 25, 22, 22, 17, 21, 15, 27, 25, 21, 24, 18, 29, 19, 19, 13, 24, 22, 12, 16, 8, 23, 25, 3, 11, 14, 14, 15, 21, 19, 37, 15, 21, 14, 20, 13, 17, 17, 14, 21, 11, 16, 19, 18, 20, 12, 18, 14, 25, 21, 16, 21, 11, 17, 25, 16, 23, 23, 21, 18, 16, 15, 23, 23, 8, 35, 18, 26, 21, 16, 17, 10, 11, 20, 9, 17, 18, 5, 23, 9, 17, 15, 26, 23, 7, 17, 22, 16, 33, 21, 9, 23, 15, 14, 21, 16, 12, 15, 8, 20, 22, 15, 18, 15, 19, 24, 19, 19, 20, 20, 12, 31, 12, 16, 17, 15, 18, 21, 20, 23, 10, 16, 27, 18, 18, 17, 25, 7, 17, 31, 22, 14, 21, 16, 19, 19, 15, 24, 15, 24, 14, 16, 17, 18, 12, 10, 20, 20, 13, 13, 43, 13, 20, 19, 18, 22, 21, 15, 8, 8, 34, 22, 16, 49, 25, 21, 20, 21, 19, 17, 18, 12, 25, 10, 14, 31, 12, 22, 20, 16, 19, 22, 24, 19, 27, 13, 9, 22, 17, 35, 28, 21, 43, 42, 14, 19, 23, 20, 15, 31, 17, 19, 17, 7, 13, 24, 13, 18, 23, 9, 19, 17, 15, 14, 8, 19, 20, 19, 27, 15, 23, 13, 11, 20, 1, 16, 11, 26, 19, 29, 20, 24, 37, 22, 22, 16, 23, 4, 14, 32, 22, 18, 24, 18, 22, 18, 19, 19, 12, 22, 19, 27, 16, 17, 16, 21, 17, 18, 8, 7, 24, 18, 11, 10, 20, 27, 31, 20, 25, 11, 23, 22, 38, 10, 6, 17, 19, 18, 16, 16, 18, 10, 19, 23, 10, 10, 19, 19, 73, 23, 20, 26, 9, 22, 17, 21, 4, 19, 29, 23, 24, 20, 20, 8, 16, 21, 18, 18, 13, 21, 17, 16, 12, 23, 16, 30, 24, 14, 2, 9, 14, 16, 21, 15, 16, 21, 20, 26, 8, 37, 25, 18, 8, 26, 21, 9, 12, 26, 33, 11, 15, 34, 13, 15, 7, 25, 17, 19, 17, 16, 30, 20, 17, 12, 23, 22, 22, 27, 19, 13, 9, 25, 30, 7, 32, 15, 28, 16, 26, 17, 21, 25, 12, 26, 20, 12, 18, 11, 22, 17, 16, 25, 16, 19, 12, 21, 31, 5, 22, 18, 40, 16, 23, 21, 5, 14, 23, 23, 17, 18, 18, 10, 12, 23, 13, 25, 16, 16, 21, 22, 20, 32, 19, 9, 20, 4, 17, 18, 18, 24, 10, 22, 4, 27, 15, 29, 9, 20, 22, 14, 21, 14, 16, 10, 18, 17, 17, 27, 5, 10, 19, 20, 14, 22, 15, 27, 26, 12, 22, 6, 18, 21, 14, 13, 14, 20, 9, 21, 13, 12, 18, 26, 13, 16, 19, 20, 9, 26, 32, 22, 20, 12, 6, 16, 24, 11, 22, 17, 21, 18, 28, 15, 33, 22, 24, 20, 19, 19, 15, 20, 18, 23, 16, 18, 21, 23, 23, 10, 23, 17, 16, 5, 30, 22, 21, 20, 19, 16, 27, 30, 28, 24, 19, 7, 29, 17, 7, 24, 20, 16, 22, 16, 26, 12, 28, 11, 10, 24, 23, 16, 19, 12, 18, 26, 17, 16, 19, 25, 15, 20, 22, 25, 18, 30, 17, 26, 33, 18, 19, 11, 20, 20, 10, 24, 14, 15, 23, 23, 20, 16, 11, 13, 20, 20, 23, 20, 18, 17, 19, 17, 23, 22, 26, 31, 5, 24, 12, 20, 16, 17, 19, 21, 23, 20, 29, 19, 24, 17, 18, 18, 18, 11, 11, 7, 24, 17, 24, 10, 19, 16, 17, 27, 36, 24, 15, 17, 2, 13, 17, 26, 15, 10, 21, 24, 15, 33, 15, 29, 17, 12, 22, 33, 19, 8, 19, 23, 22, 7, 16, 25, 17, 19, 16, 11, 15, 32, 18, 21, 23, 11, 17, 24, 30, 10, 29, 16, 19, 23, 24, 14, 8, 19, 25, 25, 18, 7, 17, 12, 35, 16, 25, 13, 24, 17, 22, 23, 21, 10, 2, 23, 15, 24, 21, 14, 15, 25, 26, 19, 25, 13, 19, 6, 21, 9, 15, 9, 13, 10, 20, 19, 32, 23, 18, 19, 24, 22, 13, 19, 19, 24, 27, 20, 21, 34, 17, 13, 10, 33, 21, 34, 19, 19, 22, 20, 21, 11, 20, 4, 7, 16, 7, 16, 18, 17, 12, 16, 22, 13, 21, 24, 11, 10, 19, 10, 5, 20, 18, 25, 21, 14, 18, 19, 23, 17, 12, 30, 18, 18, 16, 22, 27, 16, 17, 10, 16, 18, 33, 44]
      }]
  });


$('#7').highcharts({
      chart: {
          zoomType: 'x'
      },
      title: {
          text: ''
      },
      subtitle: {
          text: document.ontouchstart === undefined ?
                  'Click and drag in the plot area to zoom in' :
                  'Pinch the chart to zoom in'
      },
      xAxis: {
          type: 'datetime',
          minRange: 14 * 24 * 3600000 // fourteen days
      },
      yAxis: {
          title: {
              text: 'Happiness'
          }
      },
      legend: {
          enabled: false
      },
      plotOptions: {
          area: {
              fillColor: {
                  linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1},
                  stops: [
                      [0, Highcharts.getOptions().colors[0]],
                      [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
                  ]
              },
              marker: {
                  radius: 2
              },
              lineWidth: 1,
              states: {
                  hover: {
                      lineWidth: 1
                  }
              },
              threshold: null
          }
      },

      series: [{
          type: 'area',
          name: '',
          pointInterval: 24 * 3600 * 1000,
          pointStart: Date.UTC(2010, 0, 0),
          data: [26, 19, 31, 31, 28, 30, 29, 16, 24, 24, 13, 19, 32, 22, 16, 27, 33, 28, 23, 14, 28, 32, 17, 24, 17, 26, 13, 27, 34, 30, 25, 26, 30, 26, 29, 21, 30, 32, 28, 30, 26, 28, 33, 32, 30, 30, 23, 25, 26, 29, 19, 25, 22, 22, 26, 26, 28, 26, 40, 22, 46, 44, 45, 29, 37, 39, 26, 24, 33, 32, 29, 18, 32, 31, 22, 38, 16, 28, 33, 16, 25, 30, 30, 21, 25, 29, 23, 25, 28, 24, 37, 20, 24, 29, 19, 29, 40, 29, 29, 15, 29, 32, 29, 25, 16, 28, 22, 25, 40, 26, 28, 23, 26, 37, 19, 23, 19, 23, 27, 32, 29, 31, 19, 17, 32, 34, 31, 16, 16, 25, 22, 37, 37, 23, 22, 28, 26, 26, 32, 24, 29, 27, 37, 23, 29, 16, 33, 24, 23, 26, 33, 34, 23, 23, 23, 16, 30, 18, 38, 32, 17, 21, 26, 25, 22, 14, 33, 24, 17, 52, 38, 43, 41, 39, 26, 33, 20, 29, 27, 38, 16, 20, 30, 26, 33, 20, 29, 18, 26, 25, 32, 20, 27, 30, 41, 20, 31, 24, 21, 15, 25, 22, 19, 46, 30, 26, 22, 15, 23, 24, 27, 21, 30, 19, 21, 26, 33, 26, 19, 29, 34, 26, 31, 28, 18, 32, 18, 33, 39, 25, 25, 28, 27, 18, 35, 30, 31, 25, 27, 30, 21, 17, 27, 18, 21, 21, 29, 20, 28, 31, 33, 30, 21, 19, 34, 21, 25, 23, 43, 21, 32, 22, 24, 25, 20, 21, 28, 42, 28, 26, 30, 23, 25, 28, 22, 26, 27, 34, 24, 32, 27, 21, 33, 23, 19, 27, 21, 27, 24, 28, 29, 24, 24, 38, 25, 27, 31, 45, 21, 32, 33, 30, 29, 29, 22, 23, 33, 32, 29, 30, 29, 29, 23, 33, 32, 30, 18, 27, 31, 39, 26, 29, 33, 9, 18, 25, 16, 29, 24, 26, 35, 27, 15, 37, 25, 28, 22, 40, 21, 10, 21, 22, 22, 28, 21, 16, 26, 15, 29, 29, 14, 20, 25, 24, 15, 25, 37, 38, 32, 28, 16, 26, 37, 34, 16, 37, 37, 20, 27, 17, 26, 30, 29, 18, 24, 16, 33, 24, 24, 21, 22, 26, 15, 26, 33, 39, 22, 20, 21, 13, 18, 29, 20, 41, 23, 28, 29, 26, 27, 24, 29, 35, 36, 27, 25, 11, 40, 26, 25, 25, 26, 33, 31, 27, 26, 32, 29, 25, 27, 23, 24, 33, 44, 29, 17, 26, 25, 20, 19, 31, 19, 27, 26, 29, 26, 26, 34, 28, 28, 27, 25, 31, 27, 31, 23, 24, 16, 25, 24, 19, 21, 40, 33, 25, 24, 24, 28, 25, 15, 33, 14, 31, 25, 29, 20, 36, 21, 26, 43, 15, 15, 29, 29, 26, 25, 22, 26, 35, 33, 25, 21, 20, 16, 20, 36, 20, 24, 25, 18, 22, 25, 20, 18, 24, 33, 24, 38, 38, 18, 21, 26, 24, 25, 34, 20, 25, 27, 21, 29, 30, 22, 25, 22, 34, 19, 22, 43, 20, 25, 30, 23, 24, 19, 31, 19, 23, 22, 14, 30, 15, 43, 25, 27, 15, 16, 22, 28, 28, 29, 23, 32, 27, 25, 31, 21, 26, 26, 26, 24, 19, 22, 39, 42, 63, 46, 59, 59, 94, 26, 20, 29, 26, 36, 28, 63, 80, 49, 67, 57, 63, 68, 56, 52, 48, 44, 53, 91, 70, 65, 68, 66, 70, 36, 42, 31, 29, 57, 98, 22, 31, 22, 15, 31, 22, 13, 39, 18, 26, 16, 26, 33, 19, 36, 27, 30, 32, 12, 25, 24, 28, 23, 25, 23, 20, 28, 28, 26, 27, 29, 44, 29, 27, 54, 58, 23, 22, 26, 28, 27, 17, 23, 26, 33, 23, 24, 19, 10, 15, 26, 18, 20, 32, 21, 39, 27, 34, 19, 29, 24, 37, 23, 27, 29, 25, 26, 15, 28, 30, 22, 31, 13, 29, 27, 20, 29, 28, 25, 16, 28, 32, 18, 16, 22, 30, 30, 23, 30, 27, 18, 30, 27, 30, 35, 26, 16, 17, 26, 23, 24, 32, 29, 25, 22, 28, 26, 27, 11, 19, 18, 30, 24, 24, 35, 19, 30, 28, 19, 21, 17, 44, 32, 10, 38, 18, 24, 14, 16, 19, 29, 38, 32, 25, 16, 16, 14, 29, 22, 21, 29, 32, 23, 25, 27, 32, 22, 22, 20, 23, 40, 23, 29, 30, 25, 27, 30, 16, 24, 16, 31, 27, 28, 28, 20, 23, 30, 26, 13, 18, 22, 20, 27, 16, 22, 24, 34, 17, 22, 25, 22, 21, 22, 28, 17, 18, 26, 40, 25, 19, 32, 33, 39, 30, 19, 24, 30, 23, 31, 13, 24, 31, 27, 24, 24, 11, 31, 24, 20, 17, 28, 25, 26, 29, 28, 22, 29, 33, 20, 39, 26, 27, 25, 37, 20, 33, 20, 23, 11, 26, 27, 16, 29, 18, 71, 68, 60, 52, 32, 33, 21, 23, 45, 25, 26, 30, 33, 33, 27, 23, 37, 39, 19, 30, 23, 30, 16, 30, 26, 26, 41, 39, 29, 37, 35, 31, 27, 10, 28, 34, 27, 26, 19, 29, 19, 35, 19, 16, 33, 19, 14, 15, 23, 26, 29, 32, 28, 26, 24, 18, 17, 36, 31, 16, 25, 27, 20, 18, 26, 28, 18, 27, 40, 29, 23, 23, 29, 20, 23, 22, 24, 19, 21, 29, 25, 21, 20, 25, 30, 15, 27, 25, 24, 26, 26, 29, 28, 29, 19, 19, 29, 29, 21, 16, 22, 18, 26, 25, 23, 23, 18, 27, 34, 40, 22, 24, 30, 15, 11, 28, 9, 24, 28, 33, 26, 20, 34, 30, 17, 27, 22, 17, 32, 24, 16, 21, 22, 26, 15, 21, 13, 23, 11, 15, 27, 26, 19, 14, 27, 30, 33, 23, 24, 28, 28, 24, 47, 20, 28, 20, 13, 32, 17, 37, 22, 17, 28, 32, 26, 18, 18, 32, 24, 32, 25, 29, 33, 24, 36, 21, 18, 17, 21, 31, 24, 25, 29, 26, 19, 43, 25, 16, 32, 30, 24, 26, 20, 35, 29, 17, 23, 35, 22, 23, 28, 19, 11, 27, 24, 26, 12, 32, 23, 26, 26, 36, 32, 17, 33, 25, 34, 22, 24, 26, 33, 26, 27, 25, 27, 25, 26, 21, 33, 21, 28, 26, 31, 23, 26, 17, 20, 23, 35, 27, 28, 26, 21, 24, 18, 26, 25, 25, 33, 30, 30, 25, 29, 23, 35, 33, 29, 32, 26, 37, 27, 27, 21, 32, 30, 20, 24, 16, 31, 33, 11, 19, 22, 22, 23, 29, 27, 45, 23, 29, 22, 28, 21, 25, 25, 22, 29, 19, 24, 27, 26, 28, 20, 26, 22, 33, 29, 24, 29, 19, 25, 33, 24, 31, 31, 29, 26, 24, 23, 31, 31, 16, 43, 26, 34, 29, 24, 25, 18, 19, 28, 17, 25, 26, 13, 31, 17, 25, 23, 34, 31, 15, 25, 30, 24, 41, 29, 17, 31, 23, 22, 29, 24, 20, 23, 16, 28, 30, 23, 26, 23, 27, 32, 27, 27, 28, 28, 20, 39, 20, 24, 25, 23, 26, 29, 28, 31, 18, 24, 35, 26, 26, 25, 33, 15, 25, 39, 30, 22, 29, 24, 27, 27, 23, 32, 23, 32, 22, 24, 25, 26, 20, 18, 28, 28, 21, 21, 51, 21, 28, 27, 26, 30, 29, 23, 16, 16, 42, 30, 24, 57, 33, 29, 28, 29, 27, 25, 26, 20, 33, 18, 22, 39, 20, 30, 28, 24, 27, 30, 32, 27, 35, 21, 17, 30, 25, 43, 36, 29, 51, 50, 22, 27, 31, 28, 23, 39, 25, 27, 25, 15, 21, 32, 21, 26, 31, 17, 27, 25, 23, 22, 16, 27, 28, 27, 35, 23, 31, 21, 19, 28, 9, 24, 19, 34, 27, 37, 28, 32, 45, 30, 30, 24, 31, 12, 22, 40, 30, 26, 32, 26, 30, 26, 27, 27, 20, 30, 27, 35, 24, 25, 24, 29, 25, 26, 16, 15, 32, 26, 19, 18, 28, 35, 39, 28, 33, 19, 31, 30, 46, 18, 14, 25, 27, 26, 24, 24, 26, 18, 27, 31, 18, 18, 27, 27, 81, 31, 28, 34, 17, 30, 25, 29, 12, 27, 37, 31, 32, 28, 28, 16, 24, 29, 26, 26, 21, 29, 25, 24, 20, 31, 24, 38, 32, 22, 10, 17, 22, 24, 29, 23, 24, 29, 28, 34, 16, 45, 33, 26, 16, 34, 29, 17, 20, 34, 41, 19, 23, 42, 21, 23, 15, 33, 25, 27, 25, 24, 38, 28, 25, 20, 31, 30, 30, 35, 27, 21, 17, 33, 38, 15, 40, 23, 36, 24, 34, 25, 29, 33, 20, 34, 48, 60, 65, 59, 50, 35, 34, 33, 24, 57, 20, 29, 39, 13, 30, 26, 48, 24, 31, 29, 13, 22, 31, 31, 25, 26, 26, 18, 20, 31, 21, 33, 24, 24, 29, 30, 28, 40, 27, 17, 28, 12, 25, 26, 26, 32, 18, 30, 12, 35, 23, 37, 17, 28, 30, 22, 29, 22, 24, 18, 26, 25, 25, 35, 13, 18, 27, 28, 22, 30, 23, 35, 34, 20, 30, 14, 26, 29, 22, 21, 22, 28, 17, 29, 21, 20, 56, 44, 31, 24, 27, 28, 17, 34, 40, 30, 28, 20, 14, 24, 32, 19, 30, 25, 29, 26, 36, 23, 41, 30, 32, 28, 27, 27, 23, 28, 26, 31, 24, 26, 29, 31, 31, 18, 31, 25, 24, 13, 38, 30, 29, 28, 27, 24, 35, 38, 36, 32, 27, 15, 37, 55, 55, 52, 48, 44, 50, 54, 34, 20, 36, 19, 18, 32, 31, 24, 27, 20, 26, 64, 55, 34, 27, 33, 33, 28, 30, 33, 26, 38, 25, 34, 41, 26, 27, 19, 28, 28, 18, 32, 22, 23, 31, 31, 28, 24, 19, 21, 28, 28, 31, 28, 26, 25, 27, 25, 31, 30, 34, 39, 13, 32, 20, 28, 24, 25, 27, 29, 31, 28, 37, 27, 32, 25, 26, 26, 26, 19, 19, 15, 32, 25, 32, 18, 27, 24, 25, 35, 44, 32, 23, 25, 10, 21, 25, 34, 23, 18, 29, 32, 23, 41, 23, 37, 25, 20, 30, 41, 27, 16, 27, 31, 30, 15, 24, 33, 25, 27, 24, 19, 23, 40, 26, 29, 31, 19, 25, 32, 38, 18, 37, 24, 27, 31, 32, 22, 16, 27, 33, 33, 26, 15, 25, 20, 43, 24, 33, 21, 32, 25, 40, 41, 49, 58, 40, 31, 23, 32, 29, 22, 23, 33, 34, 27, 33, 21, 27, 14, 29, 17, 23, 17, 21, 18, 28, 27, 40, 31, 26, 27, 32, 30, 41, 37, 57, 32, 35, 28, 29, 42, 25, 21, 18, 41, 29, 42, 27, 27, 30, 28, 29, 19, 28, 12, 15, 24, 15, 24, 26, 25, 20, 24, 30, 21, 29, 32, 19, 18, 27, 18, 13, 28, 26, 33, 29, 22, 26, 27, 31, 25, 40, 38, 76, 28, 54, 30, 55, 54, 45, 38, 44, 36, 51, 62]
      }]
  });
</script>
