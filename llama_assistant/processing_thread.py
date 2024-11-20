from typing import Set
from PyQt5.QtCore import (
    QThread,
    pyqtSignal,
)
from llama_assistant.model_handler import handler as model_handler


class ProcessingThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, model, prompt, lookup_files=set(), image=None):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.image = image
        self.lookup_files = lookup_files

    def run(self):
        output = model_handler.chat_completion(
            self.model, self.prompt, image=self.image, 
            lookup_files=self.lookup_files, stream=True
        )
        full_response_str = ""
        for chunk in output:
            delta = chunk["choices"][0]["delta"]
            if "role" in delta:
                print(delta["role"], end=": ")
            elif "content" in delta:
                print(delta["content"], end="")
                full_response_str += delta["content"]
                self.update_signal.emit(delta["content"])
        model_handler.update_chat_history(full_response_str, "assistant")
        self.finished_signal.emit()


    def clear_chat_history(self):
        model_handler.clear_chat_history()
        self.finished_signal.emit()