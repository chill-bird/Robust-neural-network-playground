import gradio as gr
from datetime import datetime

import neural_net

def video():
	return gr.Video("video.mp4")

", layers, n1, n2, n3)"
def loop(lr, bs, layers, n1, n2, n3):
	neural_net.set_batch(bs)
	neural_net.set_layers(layers)
	neural_net.set_neurons1(n1)
	neural_net.set_neurons2(n2)
	neural_net.set_neurons3(n3)
	neural_net.set_policy_net()
	neural_net.set_learningRate(lr)
	neural_net.start_learning()
	return [video()]

video_if = gr.Interface(loop,
			inputs=[gr.Slider(0.001, 0.01, step=1/10**3, label="learning rate (Wert = 1/(10^learning rate)"),
                                gr.Slider(32, 1024, step=32, label="batch size (Wert = 2^batch size)"),
								gr.Slider(1, 3, step=1, label="number of hidden layers in the network"),
								gr.Slider(10, 128, step=2, label="number of nodes in first hidden layer"),
								gr.Slider(10, 128, step=2, label="number of nodes in second hidden layer"),
								gr.Slider(10, 128, step=2, label="number of nodes in third hidden layer")],
			outputs="playablevideo")

video_if.launch()
