import gradio as gr
import cv2
import os
from inference import Infer
import time

def set_chinese_font():
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

def load_model():
    try:
        model = Infer(
            'pretrained_models/epoch_00190_iteration_000400000_checkpoint.pt',
            'pretrained_models/Blender-401-00012900.pth',
            'pretrained_models/parsing.pth',
            'pretrained_models/epoch_20.pth',
            'pretrained_models/BFM'
        )
        return model
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

model = None

def head_swap(src_image, tgt_image, crop_align=True, cat=True):
    global model
    
    if model is None:
        model = load_model()
        if model is None:
            return "Model loading failed, please check if pretrained model files exist"
    
    try:
        src_path = "temp_src.jpg"
        tgt_path = "temp_tgt.jpg"
        cv2.imwrite(src_path, cv2.cvtColor(src_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(tgt_path, cv2.cvtColor(tgt_image, cv2.COLOR_RGB2BGR))
        
        start_time = time.time()
        
        result = model.run_single(src_path, tgt_path, crop_align=crop_align, cat=cat)
        
        process_time = time.time() - start_time
        print(f"Processing time: {process_time:.2f} seconds")
        
        os.remove(src_path)
        os.remove(tgt_path)
        
        if result is None:
            return "Cannot detect face in images, please try other images"
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return f"Processing failed: {str(e)}"

def create_gradio_app():
    with gr.Blocks(title="HeadSwap Tool") as app:
        gr.Markdown("## HeadSwap Tool")
        gr.Markdown("Upload source image (containing face to extract) and target image, then click Run button")
        
        with gr.Row():
            with gr.Column():
                src_image = gr.Image(label="Source Image (Face to Extract)", type="numpy")
                tgt_image = gr.Image(label="Target Image (Image to Replace)", type="numpy")
                
                crop_align = gr.Checkbox(label="Crop and Align Source Image", value=True)
                cat = gr.Checkbox(label="Show Comparison in Result", value=True)
                
                run_button = gr.Button("Run Head Swap", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Swap Result", type="numpy")
                status_text = gr.Textbox(label="Status", interactive=False)
        
        run_button.click(
            fn=head_swap,
            inputs=[src_image, tgt_image, crop_align, cat],
            outputs=[output_image]
        )
        
        gr.Markdown("### Examples")
        with gr.Row():
            gr.Markdown("Try using these example images for testing")
        
    return app

if __name__ == "__main__":
    set_chinese_font()
    app = create_gradio_app()
    
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)