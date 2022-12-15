import os
import time



from flask import Flask, request, render_template, url_for, redirect, send_from_directory, session
#from flask_mail import Mail, Message
from hockeyFinal import runAnalysis, expectedValue, getProbs
from hockeyModule import hockey_model




app = Flask(__name__)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/getProbs', methods=["POST"])
def getProb():
    homeScore = str(request.form["homeScore"])
    awayScore= str(request.form["awayScore"])
    
    isTie = 0
    homeWin = 0
    homeScore = int(homeScore)
    awayScore = int(awayScore)
    diff = abs(homeScore - awayScore)
    totalGoals = homeScore + awayScore
    homeWin = 0
    
    if homeScore > awayScore:
        homeWin = 1
    
    if homeScore == awayScore:
        isTie = 1

    
    model = hockey_model('hockeyModel', 'hockeyScaler')
    
    predictions = model.predict_one(homeScore, awayScore, diff, totalGoals, homeWin, isTie)
    
    labelPreds = model.label_predict(predictions)
    
    topPreds = model.label_top(labelPreds, 5)
    
    score1 = list(topPreds)[0]
    probs1 = list(topPreds.values())[0]
    session['probs1'] = float(probs1)
    probs1 = "{:.2%}".format(probs1)
    
    score2 = list(topPreds)[1]
    probs2 = list(topPreds.values())[1]
    session['probs2'] = float(probs2)
    probs2 = "{:.2%}".format(probs2)
    
    score3 = list(topPreds)[2]
    probs3 = list(topPreds.values())[2]
    session['probs3'] = float(probs3)
    probs3 = "{:.2%}".format(probs3)
    
    score4 = list(topPreds)[3]
    probs4 = list(topPreds.values())[3]
    session['probs4'] = float(probs4)
    probs4 = "{:.2%}".format(probs4)
    
    score5 = list(topPreds)[4]
    probs5 = list(topPreds.values())[4]
    session['probs5'] = float(probs5)
    probs5 = "{:.2%}".format(probs5)


    displayScen = "According to the neural network, the odds of 3rd period scores for a " + str(homeScore) + " to " + str(awayScore) + " game are below."    

    
    session['score1'] = score1
    session['score2'] = score2
    session['score3'] = score3
    session['score4'] = score4
    session['score5'] = score5
    
    session['homeScore'] = homeScore
    session['awayScore'] = awayScore
    session['diff'] = diff
    session['totalGoals'] = totalGoals
    session['homeWin'] = homeWin
    session['isTie'] = isTie
    
    session['displayScen'] = displayScen
    
    
    return render_template("index.html", displayScen=displayScen, score1=score1, probs1=probs1, 
                           score2=score2, probs2=probs2, score3=score3, probs3=probs3, score4=score4, 
                           probs4=probs4, score5=score5, probs5=probs5, homeScore = homeScore, awayScore = awayScore)

@app.route('/getAnalysis', methods=["POST"])
def getResults():
    probs1num = float(session.get('probs1'))
    probs1 = "{:.2%}".format(probs1num)
    
    probs2num = float(session.get('probs2'))
    probs2 = "{:.2%}".format(probs2num)
    
    probs3num = float(session.get('probs3'))
    probs3 = "{:.2%}".format(probs3num)
    
    probs4num = float(session.get('probs4'))
    probs4 = "{:.2%}".format(probs4num)
    
    probs5num = float(session.get('probs5'))
    probs5 = "{:.2%}".format(probs5num)
    
    
    score1 = session.get('score1')
    score2 = session.get('score2')
    score3 = session.get('score3')
    score4 = session.get('score4')
    score5 = session.get('score5')
    
    displayScen = session.get('displayScen')

    homeScore = session.get('homeScore')
    awayScore = session.get('awayScore')
    diff = session.get('diff')
    totalGoals = session.get('totalGoals')
    homeWin = session.get('homeWin')
    isTie = session.get('isTie')   


    model = hockey_model('hockeyModel', 'hockeyScaler')
    predictions = model.predict_one(homeScore, awayScore, diff, totalGoals, homeWin, isTie)
    labelPreds = model.label_predict(predictions)
    model.label_top(labelPreds, 5)


    
    odds1 = float(request.form["odds1"])
    odds2 = float(request.form["odds2"])
    odds3 = float(request.form["odds3"])
    odds4 = float(request.form["odds4"])
    odds5 = float(request.form["odds5"])
    bet = int(request.form["bet"])


    evResult = model.expected_value(odds1, odds2, odds3, odds4, odds5, bet)
    
    ev1 = float(evResult[0])
    ev2 = float(evResult[1])
    ev3 = float(evResult[2])
    ev4 = float(evResult[3])
    ev5 = float(evResult[4])
    
    totalScore1 = probs1num
    totalScore1 = "{:.2%}".format(totalScore1) 
    
    totalScore2 = probs1num + probs2num
    totalScore2 = "{:.2%}".format(totalScore2)

    totalScore3 = probs1num + probs2num + probs3num
    totalScore3 = "{:.2%}".format(totalScore3)
 
    totalScore4 = probs1num + probs2num + probs3num + probs4num
    totalScore4 = "{:.2%}".format(totalScore4)

    totalScore5 = probs1num + probs2num + probs3num + probs4num + probs5num
    totalScore5 = "{:.2%}".format(totalScore5)


    return render_template("index.html", probs1=probs1, probs2=probs2, probs3=probs3, probs4=probs4, probs5=probs5,
                           score1=score1, score2=score2, score3=score3, score4=score4, score5=score5,
                           displayScen = displayScen,
                           ev1=ev1, ev2=ev2, ev3=ev3, ev4=ev4, ev5=ev5,
                           odds1=odds1, odds2=odds2, odds3=odds3, odds4=odds4, odds5=odds5,
                           totalScore1=totalScore1, totalScore2=totalScore2, totalScore3=totalScore3, totalScore4=totalScore4,
                           totalScore5=totalScore5, homeScore = homeScore, awayScore = awayScore)


#app.run(host='0.0.0.0', port=80)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


#netstat -ano | findstr :80
#taskkill /PID <PID> /F