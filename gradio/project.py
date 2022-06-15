import gradio as gr

def show_picture():
	return gr.Image(value="/home/kenny/Teamprojekt/flask_app/dummy.jpg", show_label=False)

def get_numbers(ratio, noise, size):
	my_list = [ratio, noise, size]
	return my_list[0]

slider_if = gr.Interface(get_numbers,
	 	         inputs=[gr.Slider(10, 90, step="number"),
			         gr.Slider(0, 50, step="number"),
			         gr.Slider(1, 30, step="number", label="batch size")],
		         outputs="text",
		         live=True)

dummy_image_if = gr.Interface(show_picture,
	 		      inputs=[],
			      outputs=show_picture(),
			      live=True)

interface = gr.TabbedInterface([dummy_image_if, slider_if], ["Output", "Settings"])

#slider_if.launch()
#dummy_image_if.launch()
interface.launch()
