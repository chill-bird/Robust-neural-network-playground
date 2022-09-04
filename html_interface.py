from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

import neural_net


app = Flask(__name__)

@app.route("/")
def home():
	return render_template("settings.html")

@app.route("/learn")
def learn():
	lr = float(request.args.get('lr'))
	bs = int(request.args.get('bs'))
	nohl = int(request.args.get('nohl'))
	nr = int(request.args.get('nr'))

#	neural_net.set_neurons1(32)
#	neural_net.set_neurons2(32)
#	neural_net.set_neurons3(32)

	neural_net.set_batch(bs)
	neural_net.set_layers(nohl)

	if nohl == 1:
		fhl = int(request.args.get('fhl'))

		neural_net.set_neurons1(fhl)
	elif nohl == 2:
		fhl = int(request.args.get('fhl'))
		shl = int(request.args.get('shl'))

		neural_net.set_neurons1(fhl)
		neural_net.set_neurons2(shl)
	else:
		fhl = int(request.args.get('fhl'))
		shl = int(request.args.get('shl'))
		thl = int(request.args.get('thl'))

		neural_net.set_neurons1(fhl)
		neural_net.set_neurons2(shl)
		neural_net.set_neurons3(thl)

	neural_net.set_policy_net()
	neural_net.set_learningRate(lr)
	neural_net.set_nrEpisodes(nr)

	neural_net.start_learning()

	return redirect(url_for('video'))

	#return render_template("learn.html")

@app.route("/video")
def video():
	return render_template("video.html", video="./video.mp4")


if __name__ == "__main__":
        app.run(debug=True)
