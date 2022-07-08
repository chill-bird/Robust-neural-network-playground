import gradio as gr
from datetime import datetime
import neural_net

def video():
	return gr.Video("video.mp4")

def picture():
	return gr.Image("./frame.png", show_label=False)

def loop(lr, bs):
	lr = 1 / 10**lr
	bs = 2**bs
	lr = neural_net.set_learningRate(lr)
	bs = neural_net.set_batch(bs)
	test_list = [lr, bs]
	#return video()
	return

def get_numbers(ratio, noise, size):
	my_list = [ratio, noise, size]
	return my_list

frame = render.render(0, 1, 10)
frame.save("frame.png")
#loop(0, 20)

slider_if = gr.Interface(get_numbers,
	 	         inputs=[gr.Slider(10, 90, step="number"),
			         gr.Slider(0, 50, step="number"),
			         gr.Slider(1, 30, step="number", label="batch size")],
		         outputs="text",
                         live=True)

#loop_if = gr.Interface(loop,
#		       inputs=[gr.Slider(-1, 1, step="number"),
#			       gr.Slider(0, 360, step="number")],
#		       outputs=["image"],
#		       live=True)

#dummy_image_if = gr.Interface(picture,
#	 		      inputs=[gr.Slider(-1, 1, step="number"),
 #                             	      gr.Slider(0, 360, step="number")],
#			      outputs=picture())
			      #outputs=["image"],
			      #live=True)

video_if = gr.Interface(loop,
			inputs=[gr.Slider(1, 6, step="number"), #label="learning rate (Wert = 1/(10^learning rate)"),
                                gr.Slider(5, 10, step="number")], #label="batch size (Wert = 2^batch size)")],
			outputs="text")

interface = gr.TabbedInterface([video_if, slider_if], ["Output", "Settings"])

#slider_if.launch()
#dummy_image_if.launch()
interface.launch()
