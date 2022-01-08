import cv2
import detector

img = cv2.imread(f'data/mixed/test_1.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (1209, 906) if img.shape[1] > img.shape[0] else (906, 1209))

# ----- Loading Model ----- #
coin_counter = detector.Detector()
coin_counter.load_ensemble()

# ----- Detect and Classify coins ----- #
coins = coin_counter.detect(img)
coins = coin_counter.IQR(coins)

labels, labels_prob = coin_counter.classify(img, coins)
coin_counter.print_coins(img, coins, labels, labels_prob)
