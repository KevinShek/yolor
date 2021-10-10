from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob

anno_json = glob.glob('../coco/annotations/instances_val2017.json')[0]
pred_json = "runs/test/yolor_p6_onnx_transfer_learning/yolor-p6_predictions.json"
print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)

try:
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3} ] = {:0.3}'
    titleStr = ["Average Precision", "Average Precision", "Average Precision", "Average Precision", "Average Precision", "Average Precision", 
                "Average Recall", "Average Recall", "Average Recall", "Average Recall", "Average Recall", "Average Recall"]
    iouStr = ["0.50:0.95", "0.50", "0.75", "0.50:0.95", "0.50:0.95", "0.50:0.95", "0.50:0.95", "0.50:0.95", "0.50:0.95", "0.50:0.95", 
                "0.50:0.95", "0.50:0.95"]
    typeStr = ["(AP)", "(AP)", "(AP)", "(AP)", "(AP)", "(AP)", "(AR)", "(AR)", "(AR)", "(AR)", "(AR)", "(AR)"]
    areaRng = ["all", "all", "all", "small", "medium", "large", "all", "all", "all", "small", "medium", "large"]
    maxDets = ["100", "100", "100", "100", "100", "100", "1", "100", "100", "100", "100", "100"]

    map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    with open('information.txt', 'a') as f:
        for i, c in enumerate(eval.stats):
            f.write(iStr.format(titleStr[i], typeStr[i], iouStr[i], areaRng[i], maxDets[i], c) + '\n')

except Exception as e:
    print('ERROR: pycocotools unable to run: %s' % e)