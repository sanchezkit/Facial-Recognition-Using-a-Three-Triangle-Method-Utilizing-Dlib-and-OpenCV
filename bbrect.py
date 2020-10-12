import dlib


class bb_to_rect:
    def __init__(self, rect):
        self.rr = rect

    def bb_to_rect_gen(self):
        rect2 = dlib.rectangle(
            left=self.rr[0][0], top=self.rr[0][1], right=self.rr[0][2] + self.rr[0][0],
            bottom=self.rr[0][3] + self.rr[0][1]
        )
        
        return rect2
