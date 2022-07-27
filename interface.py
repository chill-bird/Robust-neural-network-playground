import gradio as gr
from datetime import datetime

import neural_net

def video():
	return gr.Video("./videos/video.mp4")

", layers, n1, n2, n3)"
def loop(lr, bs, layers, n1, n2, n3):
	lr = 1 / 10**lr
	bs = 2**bs
	neural_net.set_batch(bs)
	neural_net.set_layers(layers)
	neural_net.set_neurons1(n1)
	neural_net.set_neurons2(n2)
	neural_net.set_neurons3(n3)
	neural_net.set_policy_net()
	neural_net.set_learningRate(lr)
	test_list = [lr, bs]
	neural_net.start_learning()
	return test_list


video_if = gr.Interface(loop,
			inputs=[gr.Slider(1, 6, step="number", label="learning rate (Wert = 1/(10^learning rate)"),
                                gr.Slider(5, 10, step="number", label="batch size (Wert = 2^batch size)"),
								gr.Slider(1, 3, step="number", label="number of hidden layers in the network"),
								gr.Slider(10, 128, step="number", label="number of nodes in first hidden layer"),
								gr.Slider(10, 128, step="number", label="number of nodes in second hidden layer"),
								gr.Slider(10, 128, step="number", label="number of nodes in third hidden layer")],
			outputs="text")

video_if.launch()

