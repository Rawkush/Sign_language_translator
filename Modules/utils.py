import cv2, numpy as np
from math import ceil
'''
* These variables form part of the logic that is used for stabilizing the stream of signs
* From a stream of most recent `maxQueueSize` signs, the sign that has occured most frequently 
*   with frequency > `threshold` is considered as the consistent sign
'''
preds = []          # This is used as queue for keeping track of last `maxQueueSize` signs for finding out the consistent sign
maxQueueSize = 15   # This is the max size of queue `preds`
threshold = int(maxQueueSize/2)   # This is the minimum number of times a sign must be present in `preds` to be declared as consistent

labels_dict={'10':'a','11':'b','12':'c','13':'d','14':'e','15':'f','16':'g',
             '17':'i','18':'k','19':'l','20':'m','21':'n','22':'o','23':'p',
             '24':'q','25':'r','26':'s','27':'t','28':'u','29':'w','30':'x',
             '31':'y','32':'z'}

def getSign(inp):
    if inp<10:
        return str(inp)
    else:
        return labels_dict[str(inp)]
    
    
def get_my_hand(hand_seg_img, mask):
   
    '''
        hand_seg_img >  its coloured segmented image of hand
        mask >   black and white representation of hand
        
    '''
      
    mask1=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)#change mask to a 3 channel image 
    mask_out=cv2.subtract(mask1,hand_seg_img)
    fmask=cv2.subtract(mask1,mask_out)
    
    #finding contour using mask
    _,contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    ci = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
    if ci == -1:
        return [ False, None, None  ]
    x,y,w,h = cv2.boundingRect(contours[ci])
    # hand = np.zeros((image_skin_mask.shape[1], image_skin_mask.shape[0], 1), np.uint8)
    # cv2.drawContours(hand, contours, ci, 255, cv2.FILLED)
    # _,hand = cv2.threshold(hand[y:y+h,x:x+w], 127,255,0)
    hand_contour = fmask[y:y+h,x:x+w]
    return [ True, hand_contour, contours[ci] ]


# not needed now can be deleted
def extract_features(src_hand, grid):
    HEIGHT, WIDTH = src_hand.shape

    data = [ [0 for haha in range(grid[0])] for hah in range(grid[1]) ]
    h, w = float(HEIGHT/grid[1]), float(WIDTH/grid[0])
    
    for column in range(1,grid[1]+1):
        for row in range(1,grid[0]+1):
            fragment = src_hand[ceil((column-1)*h):min(ceil(column*h), HEIGHT),ceil((row-1)*w):min(ceil(row*w),WIDTH)]
            _,contour,_ = cv2.findContours(fragment,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            try: area = cv2.contourArea(contour[0])
            except: area=0.0
            area = float(area/(h*w))
            data[column-1][row-1] = area
    
    features = []
    for column in range(grid[1]):
        for row in range(grid[0]):
            features.append(data[column][row])
    return features


def predictSign(classifier,features):
    pred = classifier.predict([features])[0] 
    # print(pred)
    return pred


def addToQueue(pred):

    '''
    Adds the latest sign recognized to a queue of signs. This queue has maxlength: `maxQueueSize`
    Parameters
    ----------
    pred : This is the latest sign recognized by the classifier.
            This is of type number and the sign is in ASCII format

    '''
    global preds, maxQueueSize, threshold
    print("Received Sign:",pred)
    if len(preds) == maxQueueSize:
        preds = preds[1:]
    preds += [pred]
    

def getConsistentSign(displayWindows):
    '''
    From the queue of signs, this function returns the sign that has occured most frequently 
    with frequency > `threshold`. This is considered as the consistent sign.

    Returns
    -------
    number
        This is the modal value among the queue of signs.

    '''
    global preds, maxQueueSize, threshold
    modePrediction = -1
    count = threshold

    if len(preds) == maxQueueSize:
        countPredictions = {}

        for pred in preds:
            if pred != -1:
                try:
                    countPredictions[pred]+=1
                except:
                    countPredictions[pred] = 1
        
        for i in countPredictions.keys():
            if countPredictions[i]>count:
                modePrediction = i
                count = countPredictions[i]
        if displayWindows:
            displaySignOnImage(modePrediction)
    
    return modePrediction

def displayTextOnWindow(windowName,textToDisplay,xOff=75,yOff=100,scaleOfText=2):
    '''
    This just displays the text provided on the cv2 window with WINDOW_NAME: `windowName`

    Parameters
    ----------
    windowName : This is WINDOW_NAME of the cv2 window on which the text will be displayed
    textToDisplay : This is the text to be displayed on the cv2 window

    '''
    signImage = np.zeros((200,400,1),np.uint8)
    cv2.putText(signImage,textToDisplay,(xOff,yOff),cv2.FONT_HERSHEY_SIMPLEX,scaleOfText,(255,255,255),3,8);
    cv2.imshow(windowName,signImage);

def displaySignOnImage(predictSign):
    '''
    This abstracts the logic for handling signs that have not been detected in majority.
    Parameters
    ----------
    predictSign : This is the recognized sign (in ASCII) to be displayed on the cv2 window

    '''
    dispSign = "--"
    if predictSign != -1:
        dispSign = predictSign+"";  # making predictSign into a string to display on screen

    displayTextOnWindow("Prediction",dispSign)

def segment(src_img):
    """
    ### Segment skin areas from hand using a YCrCb mask.

    This function returns a mask with white areas signifying skin and black areas otherwise.

    Returns: mask
    """
    
    import cv2
    from numpy import array, uint8

    blurred_img = cv2.GaussianBlur(src_img,(5,5),0)
    blurred_img = cv2.medianBlur(blurred_img,5) 
    blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2YCrCb)
    lower = array([0,137,100], uint8)
    upper = array([255,200,150], uint8)
    mask = cv2.inRange(blurred_img, lower, upper)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask