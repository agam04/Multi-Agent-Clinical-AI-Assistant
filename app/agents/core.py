class MedicalAgentBase:
    def __init__(self, name: str):
        self.name = name

    def setup_model(self):
        raise NotImplementedError("Subclasses must implement setup_model()")

    def infer(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement infer()")
