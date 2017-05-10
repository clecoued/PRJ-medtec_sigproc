import ast
import numpy as np
from PIL import Image
import os.path

def loadReferencesImages(rawImagesFolder, groundTruthImagesFolder):
    rawImages = []
    groundTruthImages = []

    rawImagesPath = os.path.join(os.getcwd(), rawImagesFolder)
    groundTruthImagesPath = os.path.join(os.getcwd(), groundTruthImagesFolder)
    print rawImagesPath
    print groundTruthImagesPath

    validRawImageFormat = [".csv", ".gz"]
    validGroundTruthImageFormat = [".bmp"]

    # load raw images
    for f in os.listdir(rawImagesPath):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in validRawImageFormat:
            continue

        rawImage = np.loadtxt(os.path.join(rawImagesPath, f), delimiter=';')
        rawImages.append( rawImage )

    # load groundTruth images
    for f in os.listdir(groundTruthImagesPath):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in validGroundTruthImageFormat:
            continue
        # convert to grey level image and normalize intensity
        image = Image.open(os.path.join(groundTruthImagesPath, f)).convert('L')
        groundTruthImage = normImag(np.array(image))
        groundTruthImages.append( groundTruthImage )

    return {"rawImages": rawImages, "groundTruthImages": groundTruthImages}


def normImag(A):
    A = A - A.min()
    A = 1.0*A/A.max()
    return(A)

def ssd(A,B):
    A = A - 0.95*A.min()
    A = 1.0*A/A.max()
    B = B - 0.95*B.min()
    B = 1.0*B/B.max()
    squares = (A[:,:] - B[:,:]) ** 2
    return np.sum(squares)

def estimateScore(groundTruth, reconstructedImage) :
    errorMap = (groundTruth - reconstructedImage)
    score = ssd(reconstructedImage,groundTruth)
    maxErr = errorMap.max()
    return [score,maxErr]


def reconstructBaseline(rawSignal,image_shape) :
    reconstructedImage = np.zeros(shape=(image_shape[0],image_shape[1]))
    decimationFactor = 1.0*rawSignal.shape[0]/image_shape[0]

    for i in range(rawSignal.shape[0]):
           for j in range(image_shape[1]): 
                reconstructedImage[int(i/decimationFactor)][j] += np.abs(rawSignal[i][j])
                
    reconstructedImage = normImag(np.abs(reconstructedImage))
    return reconstructedImage

def execute_enveloppe():
    im = Image.open("fantom.bmp").convert('L') # convert 'L' is to get a flat image, not RGB
    groundTruth = normImag(np.array(im))
    rawSignal = np.loadtxt("SinUs.csv.gz", delimiter=';')
    recon = submit_function.reconstructImage(rawSignal,groundTruth.shape)
    [score,maxErr] = estimateScore(groundTruth, recon)
    print('Score for Baseline method : ', score)
    print('max Err between pixels for Baseline method : ', maxErr)

def execute_user_script():
    f = open("uploaded_custom.py", "r")
    data = f.read()

    tree = ast.parse(data)
    # tree.body[0] contains FunctionDef for fun1, tree.body[1] for fun2

    str_func = ""
    run_func = ""
    for function in tree.body:
        if isinstance(function,ast.FunctionDef):
            # Just in case if there are loops in the definition
            lastBody = function.body[-1]
            while isinstance (lastBody,(ast.For,ast.While,ast.If)):
                lastBody = lastBody.Body[-1]
            lastLine = lastBody.lineno
            if function.name == 'install_packages' :
                st = data.split("\n")
                for i , line in enumerate(st,1):
                    if i in range(function.lineno,lastLine+1):
                        str_func = str_func +'\n'+ line


            elif function.name == 'run':
                st = data.split("\n")
                for i , line in enumerate(st,1):
                    if i in range(function.lineno,lastLine+1):
                        run_func = run_func +'\n'+ line

    exec str_func
    exec run_func
    import time

    val_ret = {'duration':'0','score':'10000'}    
    
    imageLoader = loadReferencesImages("raw_images", "ground_truth_images")

    print len(imageLoader["rawImages"])
    print len(imageLoader["groundTruthImages"])

    rawImages = imageLoader["rawImages"]
    groundTruthImages = imageLoader["groundTruthImages"]

    if len(rawImages) != len(groundTruthImages):
        raise Exception

    #recon = submit_function.reconstructImage(rawSignal,groundTruth.shape)
    #print('Score for Baseline method : ', score)
    #print('max Err between pixels for Baseline method : ', maxErr)
    
    print 'install packages'
    install_packages()
    print 'install done'

    totalDuration = 0
    totalScore = 0
    for i in range(0, len(rawImages) ):
        print 'Process Image ' + str(i)

        start = time.clock()
        print 'execute user script'
        print str(len(rawImages)) + ' ' + str(len(groundTruthImages))

        recon = run(rawImages[i], groundTruthImages[i].shape)
        end = time.clock()

        print 'calculate score'
        [score,maxErr] = estimateScore(groundTruthImages[i], recon)

        totalDuration +=  (end - start)
        totalScore += score
        
    val_ret["duration"] = totalDuration
    val_ret["score"] = totalScore
    print val_ret
    return val_ret

