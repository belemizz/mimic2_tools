"""
 icustay class
"""

class icustay:
    def __init__(self, icustay_id, intime, outtime):
        self.icustay_id = icustay_id
        self.intime = intime
        self.outtime = outtime

    def set_medications(self, medications):
        self.medications = medications
        
    
