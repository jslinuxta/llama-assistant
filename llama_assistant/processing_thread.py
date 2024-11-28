from typing import Set, Optional, Dict
from PyQt5.QtCore import (
    QThread,
    pyqtSignal,
)
from llama_assistant.model_handler import handler as model_handler


class ProcessingThread(QThread):
    preloader_signal = pyqtSignal(str)
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        model: str,
        generation_setting: Dict,
        rag_setting: Dict,
        prompt: str,
        lookup_files: Optional[Set[str]] = None,
        image: str = None,
    ):
        super().__init__()
        self.model = model
        self.generation_setting = generation_setting
        self.rag_setting = rag_setting
        self.prompt = prompt
        self.image = image
        self.lookup_files = lookup_files
        self.preloadong = False

    def run(self):
        output = model_handler.chat_completion(
            self.model,
            self.generation_setting,
            self.rag_setting,
            self.prompt,
            image=self.image,
            lookup_files=self.lookup_files,
            stream=True,
            processing_thread=self,
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

    def emit_preloading_message(self, message: str):
        self.preloader_signal.emit(message)

    def set_preloading(self, preloading: bool, message: str):
        self.preloading = preloading
        self.emit_preloading_message(message)

    def is_preloading(self):
        return self.preloading
