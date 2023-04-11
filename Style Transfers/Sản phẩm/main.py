from fastapi import FastAPI, File, UploadFile,HTTPException, status 
from fastapi.responses import FileResponse, HTMLResponse
from style_transfer_model import StyleTransfer

app = FastAPI()

@app.get("/")
async def hello():
    html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>My App</title>
            <style>
                h1 {
                    font-size: 36px;
                }
                h2 {
                    font-size: 30px;
                }
                a {
                    font-size: 25px
                }
            </style>
        </head>
        <body>
            <center>
            <h1>THƯC TẬP A40405</h1>
            <h2>Chào mừng các bạn đến với trang web của chúng tôi, được xây dựng bằng FastAPI - 
            một framework Python siêu tốc và dễ sử dụng. Hãy trải nghiệm FastAPI 
            và cùng khám phá sức mạnh của nó nhé!</h2>
            <a href="/docs" target="_blank" style="font-weight: bold;">Link app</a>
            </center>
        </body>
        </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/style-transfer/")
async def style_transfer(file1: UploadFile = File(..., description="Style image (JPG or PNG)"), file2: UploadFile = File(..., description="Content image (JPG or PNG)")):
    # if not file1 or not file2:
    #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Bạn cần cung cấp đủ 2 tệp Style image và Content image")
    if not is_image_file(file1.filename):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tệp Style image phải là hình ảnh JPG hoặc PNG.")
    if not is_image_file(file2.filename):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tệp Content image phải là hình ảnh JPG hoặc PNG.")
    st = StyleTransfer(epochs=5,steps_per_epoch=50)
    st(file1.file, file2.file)
    st.run_gradient()
    st.save_image(img_format = 'jpg')

@app.get("/style-transfer/")
async def get_image(): 
    return FileResponse('stylized-image.jpg', media_type='image/jpeg')


def is_image_file(filename: str) -> bool:
    return filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")


# uvicorn main:app --reload