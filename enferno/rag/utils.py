import base64
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

class ImageLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Ensure OPENAI_API_KEY is available in environment
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def load(self) -> List[Document]:
        try:
            base64_image = self.encode_image(self.file_path)
            
            prompt = \"\"\"
            Please analyze this image and extract all text content faithfully. 
            If the image contains handwritten text (Arabic or English), transcribe it exactly.
            If it is a receipt, form, or structured document, represent the key-value pairs clearly.
            Do not summarize; provide the full textual content found in the image.
            \"\"\"

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ]
            )

            # Invoke the vision model
            response = self.llm.invoke([message])
            content = response.content

            return [Document(
                page_content=content,
                metadata={"source": self.file_path, "type": "image_transcription"}
            )]
        except Exception as e:
            print(f"Error processing image {self.file_path}: {e}")
            return []
