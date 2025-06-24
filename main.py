import streamlit as st
import requests
import math
import os
import plotly.express as px
import pandas as pd
import pydeck as pdk
from math import radians, sin, cos, sqrt, atan2
import time

import os
from dotenv import load_dotenv

load_dotenv()

# Ustawienia API
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY") 
SLOWO_KLUCZOWE = "restauracja"
PROMIEN = 2000  # w metrach
MAX_PHOTOS = 10  # do normalizacji bonusu za zdjęcia

# UI
st.set_page_config(page_title="Ranking Restauracji", layout="wide")
st.title("🍽️ Najlepsze Restauracje")
st.markdown("Wyszukiwanie i ranking restauracji w oparciu o ocenę, popularność, cenę, zdjęcia i inne czynniki.")

adres_startowy = st.text_input("📍 Wprowadź adres startowy (np. 'Rynek, Wrocław'):", value="Wrocław")

st.sidebar.header("Ustawienia oceny")
with st.sidebar.form("form_oceny"):
    w_rating = st.slider("Waga: Ocena", 0.0, 10.0, 4.0)
    w_reviews = st.slider("Waga: Liczba opinii", 0.0, 10.0, 3.0)
    w_price = st.slider("Waga: Cena", 0.0, 10.0, 2.0)
    w_photos = st.slider("Waga: Zdjęcia", 0.0, 10.0, 1.0)
    decay_km = st.slider("Preferowana odległość (km)", 0.1, 5.0, 2.0)
    submitted = st.form_submit_button("🔍 Szukaj restauracji")

# Funkcje pomocnicze

def nearby_distance_with_pagination(lat, lon, api_key, keyword, max_pages=5):
    """
    Pobiera kolejne strony Nearby Search z rankby=distance (do max_pages*20 wyników).
    Zwraca listę raw results.
    """
    places = []
    # pierwsze zapytanie
    url = (
        f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        f"location={lat},{lon}&rankby=distance&keyword={keyword}&key={api_key}"
    )

    for _ in range(max_pages):
        resp = requests.get(url).json()
        results = resp.get("results", [])
        places.extend(results)
        token = resp.get("next_page_token")
        if not token:
            break
        time.sleep(2)  # token needs ~2s
        url = (
            f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
            f"pagetoken={token}&key={api_key}"
        )

    return places

def generate_offset_points(lat, lon, radius_m=2000, n_points=4):
    """
    Zwraca n_points współrzędnych rozłożonych równomiernie na okręgu
    o promieniu radius_m wokół (lat, lon).
    """
    offsets = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        dy = radius_m * sin(angle)
        dx = radius_m * cos(angle)
        dlat = dy / 111000
        dlon = dx / (111000 * cos(radians(lat)))
        offsets.append((lat + dlat, lon + dlon))
    return offsets

def pobierz_100_miejsc(lat, lon, api_key, keyword):
    unique = {}
    # zawsze inklu­dujemy też oryginalny punkt
    punkt_startowe = [(lat, lon)] + generate_offset_points(lat, lon, radius_m=2000, n_points=4)

    for (plat, plon) in punkt_startowe:
        miejsca = nearby_distance_with_pagination(plat, plon, api_key, keyword)
        for m in miejsca:
            pid = m["place_id"]
            if pid not in unique:
                unique[pid] = m
            if len(unique) >= 100:
                break
        if len(unique) >= 100:
            break

    return list(unique.values())  # zwracamy listę raw results (max 100)

def geokoduj_adres(adres):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={adres}&key={API_KEY}"
    resp = requests.get(url).json()
    if resp.get("status") == "OK":
        loc = resp["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None, None


def punktuj_miejsce(miejsce, lokalizacja_uzytkownika, weights, decay_distance_km):
    # Haversine
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    # Dane miejsca
    ocena = miejsce.get("rating", 0) / 5
    opinie = miejsce.get("user_ratings_total", 1)
    price = miejsce.get("price_level", 2)
    godziny = miejsce.get("opening_hours", {})
    geo = miejsce.get("geometry", {}).get("location", {})
    photos = miejsce.get("photos", [])

    # Baza
    baza = (
        weights["rating"] * (ocena ** 1.5)
        + weights["reviews"] * math.log1p(opinie)
        - weights["price"] * abs(price - 2) ** 1.2
    )

    # Bonusy stałe
    #bonus_open = 2 if godziny.get("open_now") else 0
    #bonus_name = 1 if SLOWO_KLUCZOWE in miejsce.get("name", "").lower() else 0
    #bonus_type = 1 if any(SLOWO_KLUCZOWE in t for t in miejsce.get("types", [])) else 0
    #bonus_site = 1 if miejsce.get("website") else 0

    # Bonus zdjęć
    photo_count = len(photos)
    norm_photos = min(photo_count / MAX_PHOTOS, 1.0)
    bonus_photos = weights["photos"] * norm_photos

    # Bonus odległości
    if geo:
        d = haversine(lokalizacja_uzytkownika[0], lokalizacja_uzytkownika[1], geo.get("lat"), geo.get("lng"))
        bonus_dist = math.exp(-d / (decay_distance_km * 1000))
    else:
        bonus_dist = 0

    total = baza + bonus_photos + bonus_dist # + bonus_open + bonus_name + bonus_type + bonus_site
    return {
        "score": round(total, 2),
        "photos_count": photo_count
    }


def wyszukaj_miejsca(lat, lon):
    url = (
        f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        f"location={lat},{lon}&radius={PROMIEN}&keyword={SLOWO_KLUCZOWE}&key={API_KEY}"
    )
    return requests.get(url).json().get("results", [])


def pobierz_szczegoly(pid):
    fields = (
        "name,rating,user_ratings_total,price_level,formatted_address," \
        "opening_hours,types,website,url,geometry,photos"
    )
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={pid}&fields={fields}&key={API_KEY}"
    return requests.get(url).json().get("result", {})


def pobierz_url_zdjecia(ref, maxwidth=400):
    return (
        f"https://maps.googleapis.com/maps/api/place/photo?"
        f"maxwidth={maxwidth}&photoreference={ref}&key={API_KEY}"
    )
    
def wyswietl_nazwy(miejsca):
    """Wyświetla w Streamlit samą listę nazw miejsc."""
    st.subheader("🛠️ Nazwy pobrane z Google Places API")
    # jeśli wolisz ładny punktowany widok:
    for idx, miejsce in enumerate(miejsca, start=1):
        nazwa = miejsce.get("name", "— brak nazwy —")
        st.write(f"{idx}. {nazwa}")

# Główna logika
lat, lon = geokoduj_adres(adres_startowy)
if not lat:
    st.error("Nie udało się znaleźć lokalizacji.")
    st.stop()

if submitted:
    with st.spinner("Szukam..."):
        # miejsca = wyszukaj_miejsca(lat, lon)
        miejsca = pobierz_100_miejsc(lat, lon, API_KEY, SLOWO_KLUCZOWE)
        
        # wyswietl_nazwy(miejsca)

        
        wyniki = []
        weights = {"rating": w_rating, "reviews": w_reviews, "price": w_price, "photos": w_photos}
        for m in miejsca:
            det = pobierz_szczegoly(m["place_id"])
            data = {**m, **det}
            scores = punktuj_miejsce(data, (lat, lon), weights, decay_km)
            wyniki.append({
                "Nazwa": data.get("name"),
                "Ocena": data.get("rating"),
                "Opinie": data.get("user_ratings_total"),
                "Poziom cenowy": data.get("price_level"),
                "Adres": data.get("formatted_address"),
                "Strona WWW": data.get("website"),
                "Google Maps": data.get("url"),
                "Wynik": scores["score"],
                "Liczba zdjęć": scores["photos_count"],
                "lat": data.get("geometry", {}).get("location", {}).get("lat"),
                "lon": data.get("geometry", {}).get("location", {}).get("lng"),
                "photos": data.get("photos", [])
            })

        wyniki.sort(key=lambda x: x["Wynik"], reverse=True)
        df = pd.DataFrame(wyniki)

        # Tabela z podstawowymi danymi
        st.subheader("🔍 Podstawowe dane:")
        st.dataframe(df[["Nazwa", "Liczba zdjęć", "Ocena", "Opinie", "Poziom cenowy", "Wynik"]])

        # Wykres
        top10 = df.nlargest(10, "Wynik")  # lub df.head(10), bo już masz df posortowane
        fig = px.bar(top10, x="Nazwa", y="Wynik", hover_data=["Ocena", "Opinie", "Poziom cenowy", "Liczba zdjęć"],
                     title="📊 Wyniki restauracji", color="Wynik", color_continuous_scale="Teal")
        st.plotly_chart(fig, use_container_width=True)

        # Mapa
        st.subheader("📍 Mapa lokalizacji")
        map_df = df[["Nazwa", "lat", "lon"]].copy()
        map_df = pd.concat([map_df, pd.DataFrame([{"Nazwa": "📍 Start", "lat": lat, "lon": lon}])], ignore_index=True)
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v12",
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=13),
            layers=[pdk.Layer("ScatterplotLayer", data=map_df, get_position='[lon, lat]', pickable=True, get_radius=40, get_color='[200,30,0,160]')],
            tooltip={"text": "{Nazwa}"}
        ))

        # Zdjęcia
        st.subheader("📸 Zdjęcia restauracji")
        for _, row in df.iterrows():
            st.markdown(f"### {row['Nazwa']}")
            if row['photos']:
                ref = row['photos'][0]['photo_reference']
                st.image(pobierz_url_zdjecia(ref))
            else:
                st.warning("Brak zdjęć.")
            st.markdown(f"**Ocena:** {row['Ocena']} | **Opinie:** {row['Opinie']} | **Poziom cenowy:** {row['Poziom cenowy']}")
            st.markdown(f"[Strona WWW]({row['Strona WWW']}) | [Google Maps]({row['Google Maps']})")
    
else:
    st.info("Skonfiguruj wagi i kliknij Szukaj.")

st.markdown("---")