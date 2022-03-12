from django.http import HttpResponse
from django.shortcuts import render
from logic import score_sentence
import pandas as pd
import plotly
import plotly.express as px
def index(request):
    # return HttpResponse("Hello, world. You're at the polls index.")\
    if request.method == 'POST':
        var = request.POST['search']
        temp = score_sentence(var)
        total = 0
        for k, v in temp.items():
            pct = v * 100
            temp[k] = pct
            total += pct
        print(total)
        df = pd.DataFrame(temp.items(), columns=['Emotion', 'Percent'])
        print(df)
        fig = px.pie(df, values = "Percent", names = "Emotion", title = "Twitter Emotion")
        fig.update_traces(hoverinfo='label+percent+name', textinfo='none')
        fig.update_layout(plot_bgcolor='rgb(18,18,18)',paper_bgcolor ='rgb(18,18,18)', font_color = 'rgb(255,255,255)')
        pie = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
    return render(request, "index.html", context={'plot_div': pie})
