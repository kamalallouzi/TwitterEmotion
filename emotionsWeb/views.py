from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from logic import score_sentence
import tweepy
from tweepy import API, Client, Paginator, Cursor, OAuthHandler
import pandas as pd
import plotly
import plotly.express as px
import json
def index(request):
    # return HttpResponse("Hello, world. You're at the polls index.")
    try:
        if request.method == 'POST':
            finalDict = {
                    "admiration":0,          
                    "amusement":0,
                    "anger":0,
                    "annoyance":0,
                    "approval":0,
                    "caring":0,
                    "confusion":0,
                    "curiosity":0,
                    "desire":0,
                    "disappointment":0,
                    "disapproval":0,
                    "disgust":0,
                    "embarrassment":0,
                    "excitement":0,
                    "fear":0,
                    "gratitude":0,
                    "grief":0,
                    "joy":0,
                    "love":0,
                    "nervousness":0,
                    "optimism":0,
                    "pride":0,
                    "realization":0,
                    "relief":0,
                    "remorse":0,
                    "sadness":0,
                    "surprise":0,
                    "neutral":0
            }
            numSentences = 0
            var = request.POST['search']
            consumer_key="X7PHeTECbzqqWUMlJ5PVb0RfZ"
            consumer_secret="jdfjO7gYg3Ph2S94By56CO5bViYDpG2cBoGBWiFFdWJRcwe9Iy"
            access_token="1503417943073312773-SlMGAdu2BL6h10VPMHO36aVBRWWzQb"
            access_token_secret="mSTEfjO4213yyVJGX8cbvVxAybJCF5V9yMlZTbKF6GDSc"
            client = Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAMjpaAEAAAAAmaitCSGU7BakigfPCpwOkQjlNok%3DmAQPrE7BjlUgt5GuLv4Caa3BQq9vTtNAGBEsuPKWmX17p6mKjk', consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token, access_token_secret=access_token_secret)
            user = client.get_user(username = var)
            tweets = Paginator(client.get_users_tweets, id=user.data.id, tweet_fields=['context_annotations','created_at'], max_results=100).flatten(limit=100)
            for tweet in tweets:
                numSentences += 1
                temp = score_sentence(str(tweet))
                total = 0
                for k, v in temp.items():
                    pct = v * 100
                    finalDict[k] += pct
            for k, v in finalDict.items():
                    finalDict[k] = finalDict[k] / numSentences
                    # print(finalDict[k])
            df = pd.DataFrame(finalDict.items(), columns=['Emotion', 'Percent'])
            fig = px.pie(df, values = "Percent", names = "Emotion", title = "Twitter Emotion")
            fig.update_traces(hoverinfo='label+percent+name', textinfo='none')
            fig.update_layout(plot_bgcolor='rgb(18,18,18)',paper_bgcolor ='rgb(18,18,18)', font_color = 'rgb(255,255,255)')
            pie = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
            data = []
            json_records = df.reset_index().to_json(orient ='records')
            data = json.loads(json_records)
            return render(request, "index.html", context={'plot_div': pie, "d": data})
    except AttributeError:
        messages.success(request, "Please input a valid username!")
        return redirect('/')
    except tweepy.errors.HTTPException:
        messages.success(request, "Please input a valid username!")
        return redirect('/')
    except ZeroDivisionError:
        messages.success(request, "This user does not have any tweets!")
        return redirect('/')
    return render(request, "index.html")
