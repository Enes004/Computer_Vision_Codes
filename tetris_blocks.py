import cv2
import numpy as np

# 1. Resmi oku ve RGB'ye çevir
image = cv2.imread('/home/enes/Documents/github_repos/Computer_Vision_Codes/Enes Satıcı - 23 Ara 2025, 22_02.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Renkler ve sınırları (Dictionary)
# Her renk için [Kirmizi, Yesil, Mavi] alt ve üst sınırlarını tanımlıyoruz
renk_sozlugu = {
    "Kirmizi": ([150, 0, 0], [255, 80, 80]),
    "Mavi":    ([0, 0, 150], [100, 100, 255]),
    "Yesil":   ([0, 100, 0], [150, 255, 150]),   # Alt sınırı 150'den 100'e çektik ki daha koyu yeşilleri de alsın
    "Sari":    ([210, 210, 0], [255, 255, 150]), # Alt sınırı yükselttik, böylece Turuncu'yu kapsamayacak
    "Turuncu": ([200, 100, 0], [255, 180, 80]),  # Yeşili 180 ile sınırladık ki Sarı ile karışmasın
    "Turkuaz": ([0, 150, 150], [180, 255, 255])
}

# 3. Her rengi tek tek kontrol edecek döngü
for renk_adi, (alt, ust) in renk_sozlugu.items():
    
    # Listeleri numpy dizisine çeviriyoruz
    alt_sinir = np.array(alt)
    ust_sinir = np.array(ust)
    
    # O anki renk için MASKE oluştur
    maske = cv2.inRange(image_rgb, alt_sinir, ust_sinir)
    
    # O maske içindeki ŞEKİLLERİ (Konturları) bul
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for kontur in konturlar:
        # Alanı hesapla
        M = cv2.moments(kontur)
        
        # Eğer alan 200 pikselden büyükse (küçük gürültüleri elemek için)
        if M["m00"] > 200:
            # Merkezi hesapla
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Terminale sonuçları yazdır
            print(f"{renk_adi} Bloğu bulundu: X={cX}, Y={cY}")
            
            # RESME İŞARET KOY (Döngünün içinde!)
            # Her renk bloğuna siyah bir nokta koyalım
            cv2.circle(image, (cX, cY), 5, (0, 0, 0), -1)
            # Üzerine rengin adını ve koordinatlarını yazalım
            cv2.putText(image, f"{renk_adi} ({cX},{cY})", (cX - 40, cY - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

# 4. Pencereleri Göster
cv2.namedWindow('Tum Renkler Tespit Edildi', cv2.WINDOW_NORMAL)
cv2.imshow('Tum Renkler Tespit Edildi', image)

cv2.waitKey(0)
cv2.destroyAllWindows()