import gradio as gr

def picture():
	return gr.Image("frame.png", show_label=False)

def get_numbers(ratio, noise, size):
	my_list = [ratio, noise, size]
	return my_list

slider_if = gr.Interface(get_numbers,
	 	         inputs=[gr.Slider(10, 90, step="number"),
			         gr.Slider(0, 50, step="number"),
			         gr.Slider(1, 30, step="number", label="batch size")],
		         outputs="text",
                         live=True)

dummy_image_if = gr.Interface(picture,
	 		      inputs=[],
			      outputs=picture())

interface = gr.TabbedInterface([dummy_image_if, slider_if], ["Output", "Settings"])

#slider_if.launch()
#dummy_image_if.launch()
interface.launch()
