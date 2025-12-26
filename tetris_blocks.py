import cv2
import numpy as np

#Read the img and covert it RGB
image = cv2.imread('/home/enes/Documents/github_repos/Computer_Vision_Codes/Enes Satıcı - 23 Ara 2025, 22_02.jpg')
image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


print(image.shape)

#Derive lower and upper ranges for red
lower_red =np.array([100,0,0])
upper_red = np.array([255,120,120])

#Convert pixels to white in this range,others are black
#MASK is only consist of white and black
mask = cv2.inRange(image_rgb,lower_red,upper_red)

# Contours (borders of shape)
#Şeklin Sınırlarını Bulma (cv2.findContours)
contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
'''
cv2.RETR_EXTERNAL: "Sadece en dış sınırları bul" demektir. Eğer bloğun içinde küçük delikler varsa onları görmezden gelmesini sağlar.
cv2.CHAIN_APPROX_SIMPLE: "Hafızayı tasarruflu kullan" demektir. Bir karenin tüm kenarlarındaki binlerce noktayı kaydetmek yerine, sadece 4 köşe noktasını kaydeder.
konturlar: Her bir sınır, o şeklin etrafındaki noktaların (x,y) koordinatlarını tutar..
'''

#Matematiksel Merkez Hesaplama (cv2.moments)
for contour in contours:
    #Area
    M = cv2.moments(contour)
    # if area is not aqual 0 (its a real shape)
    if M["m00"] != 0:
        # Center X = Total X / Total area
        cX = int(M["m10"] / M["m00"])
        #Center Y = Total Y / Total area
        cY = int(M["m01"]/ M["m00"])


#Koordinatları Resim Üzerinde İşaretleme
#Bulduğumuz bu (cX,cY) noktasına bir nokta koyup, yanına da "Merkez" yazdıralım.

#Merkeze bir daire çizelim
cv2.circle(image , (cX,cY), 7 ,(0,255,0), -1)

# Yanına koordinatlarını yazdıralım
cv2.putText(image, f"({cX}, {cY})", (cX - 20, cY - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



# Pencereleri Ayarla (KÜÇÜK EKRAN ÇÖZÜMÜ)
# 'WINDOW_NORMAL' komutu, pencereyi mouse ile tutup büyütmeni sağlar.
cv2.namedWindow('Orijinal Resim', cv2.WINDOW_NORMAL)
cv2.namedWindow('Maske Filtresi', cv2.WINDOW_NORMAL)
# Göster
cv2.imshow('Orijinal Resim', image)
cv2.imshow('Maske Filtresi', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()