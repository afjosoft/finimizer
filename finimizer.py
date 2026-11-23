#!/usr/bin/env python3
"""
Finimizer — optimizador de aletas trapezoidales con OpenRocket.

Requisitos:
    Python 3.10+, JDK 17 o 21
    pip install orlab numpy scipy pandas
    python -m orlab fetch

Uso:
    python finimizer.py explorar  [archivo.ork]   muestreo del espacio de diseño
    python finimizer.py optimizar [archivo.ork]   evolución diferencial + refinamiento
    python finimizer.py completo  [archivo.ork]   explora y luego optimiza desde los mejores

El programa:
    1. Abre el .ork y encuentra el set de aletas trapezoidales y su fuselaje porta-aletas.
    2. Pide los parámetros de diseño (Enter = valor por defecto, según la IREC DTEG).
    3. Estima el tiempo de la corrida y pide confirmación.
    4. Guarda los resultados en CSV y el mejor diseño en <archivo>_finimized.ork.

Variables de diseño:
    cuerda raíz, cuerda punta, envergadura, flecha, espesor y posición de las aletas
    a lo largo del fuselaje. La punta nunca sobresale del final del fuselaje.
    Restricciones opcionales de forma: cuerda punta mínima y ángulo máximo de flecha.
"""

import os
import re
import sys
import time
import tempfile
import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

import orlab
from orlab import FlightDataType as T, FlightEvent as E


# =============================================================================
# 1. CONFIGURACIÓN FIJA
# =============================================================================
ARCHIVO_ORK = "or_test.ork"
INDICE_SIMULACION = 0
FRACCION_V_VUELO = 0.3   # la estabilidad se evalúa mientras V > 30 % de Vmax
POBLACION_DE = 10        # multiplicador de población de la evolución diferencial
ITER_REFINAMIENTO = 60   # evaluaciones máximas del refinamiento local

# Reglas para los límites automáticos (múltiplos del diámetro D del fuselaje porta-aletas)
FACTOR_CUERDA_RAIZ = (0.8, 2.5)
FACTOR_ENVERGADURA = (0.5, 1.5)
FRACCION_TUBO_MAX = 0.9              # la cuerda raíz no supera el 90 % del largo del fuselaje

# Espesor: no hay límite normativo; el mínimo y el máximo los ingresa el usuario (obligatorio).

# Valores de la IREC DTEG (rev. 2024 v1.1.3). Verificar contra la versión vigente.
DTEG_ESTAB_MIN = 1.5            # cal, 10.3.1
DTEG_ESTAB_MAX_ESTATICA = 4.0   # cal, 10.4.1
DTEG_ESTAB_MAX_DINAMICA = 6.0   # cal, 10.4.1
DTEG_V_RIEL_RECOMENDADA = 30.0  # m/s, 10.2.1
DTEG_V_RIEL_ABSOLUTA = 15.0     # m/s, 10.2.2.1
DTEG_FS_FLUTTER = 1.5           # 8.2.2

DEFECTO_APOGEO = 3292           # m (≈10 800 ft: 10k ft + 8 % para airbrakes)
DEFECTO_N_EXPLORAR = 300
DEFECTO_N_GENERACIONES = 50
G0 = 9.80665                    # m/s²

# Materiales: (nombre, G en GPa). Valores aproximados: verificar con la ficha técnica.
MATERIALES = [
    ("Fibra de carbono tejida 0/90", 4.0),
    ("Fibra de carbono cuasi-isotrópica (0/±45/90)", 15.0),
    ("Fibra de vidrio tejida (G10/FR4)", 3.0),
    ("Aluminio 6061-T6", 26.0),
]
MATERIAL_DEFECTO = 1


# =============================================================================
# 2. VARIABLES GLOBALES
# =============================================================================
LIMITES = None
APOGEO_OBJETIVO = None
ESTAB_MIN = None
ESTAB_MAX = None
V_RIEL_MIN = None
FS_FLUTTER_MIN = None
G_MATERIAL = None
N_ALETAS = None
DESPLAZAMIENTO_MAX = 0.0   # m que las aletas pueden correrse hacia adelante (automático)
LARGO_FUSELAJE = None
CUERDA_PUNTA_MIN = 0.0     # m (0 = sin límite: permite aletas en pico)
FLECHA_ANG_MAX = None      # grados (None = sin límite)
N_EXPLORAR = DEFECTO_N_EXPLORAR
N_GENERACIONES = DEFECTO_N_GENERACIONES

orl = None
sim = None
aletas = None
historial = []
cache = {}


# =============================================================================
# 3. LECTURA DEL COHETE Y LÍMITES
# =============================================================================
def buscar_aletas(componente):
    if componente.getClass().getSimpleName() == "TrapezoidFinSet":
        return componente
    for hijo in componente.getChildren():
        encontrado = buscar_aletas(hijo)
        if encontrado is not None:
            return encontrado
    return None


def clase_java(*nombres):
    """Devuelve la primera clase Java que exista (los paquetes cambian entre versiones)."""
    from jpype import JClass
    for nombre in nombres:
        try:
            return JClass(nombre)
        except Exception:
            continue
    return None


def calcular_limites(diametro, largo):
    D = diametro
    cr_max = min(FACTOR_CUERDA_RAIZ[1] * D, FRACCION_TUBO_MAX * largo)
    cr_min = min(FACTOR_CUERDA_RAIZ[0] * D, 0.5 * cr_max)
    return [
        (cr_min, cr_max),                                        # cuerda raíz [m]
        (0.0, 1.0),                                              # fracción de cuerda punta [-]
        (FACTOR_ENVERGADURA[0] * D, FACTOR_ENVERGADURA[1] * D),  # envergadura [m]
        (0.0, 1.0),                                              # fracción de flecha [-]
        None,                                                    # espesor [m]: lo define el usuario
    ]


# =============================================================================
# 4. ENTRADA DEL USUARIO
# =============================================================================
def pedir_numero(mensaje, defecto=None, minimo=None, maximo=None):
    """Pide un número. Con defecto=None la respuesta es obligatoria."""
    while True:
        sufijo = f" [{defecto}]" if defecto is not None else ""
        texto = input(f"{mensaje}{sufijo}: ").strip().replace(",", ".")
        if texto == "":
            if defecto is not None:
                return defecto
            print("  Este valor es obligatorio.")
            continue
        try:
            valor = float(texto)
        except ValueError:
            print("  Ingresa un número válido.")
            continue
        if minimo is not None and valor < minimo:
            print(f"  Debe ser mayor o igual a {minimo}.")
            continue
        if maximo is not None and valor > maximo:
            print(f"  Debe ser menor o igual a {maximo}.")
            continue
        return valor


def pedir_opcional(mensaje, minimo=None, maximo=None):
    """Pide un número; Enter devuelve None (sin límite)."""
    while True:
        texto = input(f"{mensaje} [Enter = sin límite]: ").strip().replace(",", ".")
        if texto == "":
            return None
        try:
            valor = float(texto)
        except ValueError:
            print("  Ingresa un número válido.")
            continue
        if minimo is not None and valor < minimo:
            print(f"  Debe ser mayor o igual a {minimo}.")
            continue
        if maximo is not None and valor > maximo:
            print(f"  Debe ser menor o igual a {maximo}.")
            continue
        return valor


def elegir_material():
    print("\nMaterial de las aletas:")
    for i, (nombre, g) in enumerate(MATERIALES, start=1):
        marca = "  (por defecto)" if i == MATERIAL_DEFECTO else ""
        print(f"  {i}. {nombre:<46} G = {g:5.1f} GPa{marca}")
    otro = len(MATERIALES) + 1
    print(f"  {otro}. Otro (ingresar G manualmente)")
    opcion = int(pedir_numero("Opción", MATERIAL_DEFECTO, minimo=1, maximo=otro))
    if opcion == otro:
        nombre = input("  Nombre del material: ").strip() or "Material personalizado"
        g = pedir_numero("  Módulo de cortante G [GPa]", 4.0, minimo=0.1)
        return nombre, g
    return MATERIALES[opcion - 1]


def revisar_dteg():
    avisos = []
    if ESTAB_MIN < DTEG_ESTAB_MIN:
        avisos.append(f"Estabilidad mínima {ESTAB_MIN} < {DTEG_ESTAB_MIN} cal: incumple DTEG 10.3.1.")
    if ESTAB_MAX > DTEG_ESTAB_MAX_DINAMICA:
        avisos.append(f"Estabilidad máxima {ESTAB_MAX} > {DTEG_ESTAB_MAX_DINAMICA} cal: "
                      "incumple DTEG 10.4.1 (sobreestabilidad dinámica).")
    elif ESTAB_MAX > DTEG_ESTAB_MAX_ESTATICA:
        avisos.append(f"Estabilidad máxima {ESTAB_MAX} > {DTEG_ESTAB_MAX_ESTATICA} cal: verifica "
                      "que el margen ESTÁTICO no supere 4 cal (DTEG 10.4.1).")
    if V_RIEL_MIN < DTEG_V_RIEL_ABSOLUTA:
        avisos.append(f"Velocidad de riel {V_RIEL_MIN} < {DTEG_V_RIEL_ABSOLUTA} m/s: "
                      "prohibido por DTEG 10.2.2.1.")
    elif V_RIEL_MIN < DTEG_V_RIEL_RECOMENDADA:
        avisos.append(f"Velocidad de riel {V_RIEL_MIN} < {DTEG_V_RIEL_RECOMENDADA} m/s: DTEG 10.2.2 "
                      "exige análisis detallado o múltiples simulaciones que prueben la estabilidad.")
    if FS_FLUTTER_MIN < DTEG_FS_FLUTTER:
        avisos.append(f"FS flutter {FS_FLUTTER_MIN} < {DTEG_FS_FLUTTER}: incumple DTEG 8.2.2.")
    if avisos:
        print("\n*** AVISOS IREC DTEG ***")
        for a in avisos:
            print("  - " + a)
    else:
        print("\n  Valores dentro de los límites de la IREC DTEG.")


def pedir_parametros(modo):
    global APOGEO_OBJETIVO, ESTAB_MIN, ESTAB_MAX, V_RIEL_MIN, FS_FLUTTER_MIN, G_MATERIAL
    global N_ALETAS, DESPLAZAMIENTO_MAX, N_EXPLORAR, N_GENERACIONES

    print("\n=== Parámetros de diseño (Enter = valor por defecto) ===")
    print("(Valores por defecto según la IREC DTEG)")
    APOGEO_OBJETIVO = pedir_numero("Apogeo objetivo [m]", DEFECTO_APOGEO, minimo=30)
    ESTAB_MIN = pedir_numero("Estabilidad mínima durante el vuelo [cal]", DTEG_ESTAB_MIN, minimo=0.5)
    ESTAB_MAX = pedir_numero("Estabilidad máxima durante el vuelo [cal]",
                             max(DTEG_ESTAB_MAX_ESTATICA, ESTAB_MIN + 0.5), minimo=ESTAB_MIN + 0.1)
    V_RIEL_MIN = pedir_numero("Velocidad mínima de salida del riel [m/s]", DTEG_V_RIEL_RECOMENDADA,
                              minimo=3)
    FS_FLUTTER_MIN = pedir_numero("Factor de seguridad mínimo contra flutter", DTEG_FS_FLUTTER,
                                  minimo=1.0)
    revisar_dteg()

    # Espesor
    t_min_mm = pedir_numero("\nEspesor mínimo de las aletas [mm] (obligatorio)", minimo=0.1)
    t_max_mm = pedir_numero("Espesor máximo de las aletas [mm] (obligatorio)", minimo=t_min_mm)
    if t_max_mm == t_min_mm:
        t_max_mm = t_min_mm + 0.001   # espesor fijo (rango mínimo para el optimizador)
        print(f"  Espesor fijo en {t_min_mm} mm.")
    LIMITES[4] = (t_min_mm / 1000, t_max_mm / 1000)

    # Restricciones de forma
    global CUERDA_PUNTA_MIN, FLECHA_ANG_MAX
    cr_min, cr_max = LIMITES[0]
    ct_mm = pedir_opcional("\nCuerda punta mínima [mm]", minimo=0.0,
                           maximo=round(0.9 * cr_max * 1000, 1))
    CUERDA_PUNTA_MIN = (ct_mm or 0.0) / 1000
    if CUERDA_PUNTA_MIN >= cr_min:   # la raíz debe poder ser mayor que la punta mínima
        LIMITES[0] = (1.05 * CUERDA_PUNTA_MIN, cr_max)
    FLECHA_ANG_MAX = pedir_opcional("Ángulo máximo de flecha del borde de ataque [°]",
                                    minimo=0.0, maximo=85.0)

    # Posición a lo largo del fuselaje: automática, en todo el largo disponible
    if AXIAL_BOTTOM is not None:
        DESPLAZAMIENTO_MAX = LARGO_FUSELAJE
        LIMITES.append((0.0, 1.0))   # fracción del desplazamiento disponible

    nombre_material, g_gpa = elegir_material()
    G_MATERIAL = g_gpa * 1e9

    n_actual = int(aletas.getFinCount())
    while True:
        n = int(pedir_numero("\nNúmero de aletas (3 o 4)", n_actual if n_actual in (3, 4) else 4))
        if n in (3, 4):
            break
        print("  Solo se permite 3 o 4.")
    N_ALETAS = n
    aletas.setFinCount(N_ALETAS)

    # Cantidad de simulaciones
    if modo in ("explorar", "completo"):
        N_EXPLORAR = int(pedir_numero("\nNúmero de diseños a explorar", DEFECTO_N_EXPLORAR, minimo=5))
    if modo in ("optimizar", "completo"):
        N_GENERACIONES = int(pedir_numero("Número de generaciones de optimización",
                                          DEFECTO_N_GENERACIONES, minimo=1))

    material_ork = aletas.getMaterial()
    print("\nResumen:")
    print(f"  Apogeo objetivo   : {APOGEO_OBJETIVO:,.0f} m")
    print(f"  Estabilidad       : {ESTAB_MIN} – {ESTAB_MAX} cal")
    print(f"  V salida riel mín.: {V_RIEL_MIN:.1f} m/s")
    print(f"  FS flutter mín.   : {FS_FLUTTER_MIN}")
    print(f"  Espesor           : {LIMITES[4][0] * 1000:.2f} – {LIMITES[4][1] * 1000:.2f} mm")
    print(f"  Cuerda punta mín. : {CUERDA_PUNTA_MIN * 1000:.1f} mm"
          + ("  (sin límite)" if CUERDA_PUNTA_MIN == 0 else ""))
    print(f"  Flecha máx.       : "
          + (f"{FLECHA_ANG_MAX:.0f}°" if FLECHA_ANG_MAX is not None else "sin límite"))
    print(f"  Posición aletas   : {'variable a lo largo del fuselaje' if DESPLAZAMIENTO_MAX > 0 else 'fija'}")
    print(f"  Material (flutter): {nombre_material} (G = {g_gpa} GPa)")
    print(f"  Material en .ork  : {material_ork.getName()} "
          f"(densidad {material_ork.getDensity():.0f} kg/m³)")
    print(f"  Número de aletas  : {N_ALETAS}")
    print("  Nota: el material del .ork define la masa; el del menú, la rigidez. Deben coincidir.")


# =============================================================================
# 5. CÁLCULOS
# =============================================================================
def convertir_variables(x):
    """Vector del optimizador -> geometría real. Toda combinación es válida por construcción."""
    cr = x[0]
    ct = CUERDA_PUNTA_MIN + x[1] * (cr - CUERDA_PUNTA_MIN)   # entre la punta mínima y la raíz
    s = x[2]
    t = x[4]
    frac_pos = x[5] if len(x) > 5 else 0.0
    # Desplazamiento hacia adelante, sin que el borde de ataque de la raíz salga del fuselaje
    d = frac_pos * min(DESPLAZAMIENTO_MAX, max(LARGO_FUSELAJE - cr, 0.0))
    # Flecha máxima: la punta no pasa del final del fuselaje ni del ángulo máximo
    sw_max = cr + d - ct
    if FLECHA_ANG_MAX is not None:
        sw_max = min(sw_max, s * np.tan(np.radians(FLECHA_ANG_MAX)))
    sw = x[3] * sw_max
    return cr, ct, s, sw, t, d


def velocidad_flutter(a, P, cr, ct, s, t):
    """Velocidad de flutter (NACA TN 4197, forma de Apogee Peak of Flight #291)."""
    area = 0.5 * (cr + ct) * s
    AR = s**2 / area
    lam = ct / cr
    tc = t / cr
    denominador = 1.337 * AR**3 * P * (lam + 1) / (2 * (AR + 2) * tc**3)
    return a * np.sqrt(G_MATERIAL / denominador)


AXIAL_BOTTOM = None   # se carga en main()


def aplicar_geometria(cr, ct, s, sw, t, d):
    aletas.setRootChord(cr)
    aletas.setTipChord(ct)
    aletas.setHeight(s)
    aletas.setSweep(sw)
    aletas.setThickness(t)
    if AXIAL_BOTTOM is not None:
        aletas.setAxialMethod(AXIAL_BOTTOM)
        aletas.setAxialOffset(-d)   # negativo = hacia adelante desde el final del fuselaje


def simular(cr, ct, s, sw, t, d):
    aplicar_geometria(cr, ct, s, sw, t, d)
    orl.run_simulation(sim)

    tipos = [T.TYPE_TIME, T.TYPE_ALTITUDE, T.TYPE_VELOCITY_TOTAL, T.TYPE_STABILITY,
             T.TYPE_AIR_PRESSURE, T.TYPE_SPEED_OF_SOUND, T.TYPE_ACCELERATION_TOTAL]
    datos = orl.get_timeseries(sim, tipos)
    tiempo, altitud, velocidad, estabilidad, presion, vel_sonido, aceleracion = (
        np.array(datos[k], dtype=float) for k in tipos)

    t_riel = float(orl.get_events(sim)[E.LAUNCHROD][0])
    i_apogeo = int(np.nanargmax(altitud))
    indices = np.arange(len(tiempo))
    validos = np.isfinite(estabilidad)
    vuelo = ((tiempo >= t_riel) & (indices <= i_apogeo)
             & (velocidad > FRACCION_V_VUELO * np.nanmax(velocidad)))
    estab_vuelo = estabilidad[vuelo & validos]
    v_flutter = velocidad_flutter(vel_sonido[vuelo], presion[vuelo], cr, ct, s, t)

    return {
        "apogeo": altitud[i_apogeo],
        "v_max": np.nanmax(velocidad),
        "a_max_g": np.nanmax(aceleracion) / G0,
        "v_riel": np.interp(t_riel, tiempo, velocidad),
        "estab_riel": np.interp(t_riel, tiempo[validos], estabilidad[validos]),
        "estab_min": np.min(estab_vuelo),
        "estab_max": np.max(estab_vuelo),
        "fs_flutter": np.min(v_flutter / velocidad[vuelo]),
    }


def calcular_penalizacion(r):
    p = 0.0
    p += max(0, (ESTAB_MIN - r["estab_min"]) / ESTAB_MIN)
    p += max(0, (r["estab_max"] - ESTAB_MAX) / ESTAB_MAX)
    p += max(0, (V_RIEL_MIN - r["v_riel"]) / V_RIEL_MIN)
    p += max(0, (FS_FLUTTER_MIN - r["fs_flutter"]) / FS_FLUTTER_MIN)
    return p


def evaluar(x):
    x = np.clip(np.asarray(x, dtype=float), [l[0] for l in LIMITES], [l[1] for l in LIMITES])
    clave = tuple(np.round(x, 6))
    if clave in cache:
        return cache[clave]

    cr, ct, s, sw, t, d = convertir_variables(x)
    try:
        r = simular(cr, ct, s, sw, t, d)
    except Exception as e:
        print("  simulación fallida:", e)
        cache[clave] = 1e6
        return 1e6

    penalizacion = calcular_penalizacion(r)
    puntaje = abs(r["apogeo"] - APOGEO_OBJETIVO) + 1e4 * penalizacion
    fila = {"cr": cr, "ct": ct, "s": s, "sw": sw, "t": t, "d": d,
            "ang_flecha": np.degrees(np.arctan2(sw, s)), **r,
            "factible": penalizacion == 0, "penalizacion": penalizacion,
            "puntaje": puntaje, "x": x.tolist()}
    historial.append(fila)
    cache[clave] = puntaje
    print(f"  #{len(historial):4d}  apogeo={r['apogeo']:6,.0f} m  "
          f"estab={r['estab_min']:.2f}–{r['estab_max']:.2f}  "
          f"v_riel={r['v_riel']:5.1f} m/s  fs_flutter={r['fs_flutter']:.2f}  "
          f"{'OK' if fila['factible'] else '--'}")
    return puntaje


# =============================================================================
# 6. RESULTADOS
# =============================================================================
def mostrar_diseno(f, titulo="Mejor diseño"):
    print(f"\n=== {titulo} ===")
    print(f"Cuerda raíz    : {f['cr'] * 1000:.1f} mm")
    print(f"Cuerda punta   : {f['ct'] * 1000:.1f} mm")
    print(f"Envergadura    : {f['s'] * 1000:.1f} mm")
    print(f"Flecha         : {f['sw'] * 1000:.1f} mm ({f['ang_flecha']:.1f}°)")
    print(f"Espesor        : {f['t'] * 1000:.2f} mm")
    print(f"Posición       : {f['d'] * 1000:.1f} mm adelante del final del fuselaje")
    print(f"Apogeo         : {f['apogeo']:,.0f} m")
    print(f"Error apogeo   : {f['apogeo'] - APOGEO_OBJETIVO:+,.0f} m")
    print(f"Estab. vuelo   : {f['estab_min']:.2f} – {f['estab_max']:.2f} cal")
    print(f"Estab. riel    : {f['estab_riel']:.2f} cal")
    print(f"V salida riel  : {f['v_riel']:.1f} m/s")
    print(f"V máxima       : {f['v_max']:.1f} m/s")
    print(f"Aceleración máx: {f['a_max_g']:.1f} g")
    print(f"FS flutter     : {f['fs_flutter']:.2f}")


def mostrar_incumplimientos(f):
    print("\nRestricciones que incumple:")
    if f["estab_min"] < ESTAB_MIN:
        print(f"  - Estabilidad mínima: {f['estab_min']:.2f} cal (requerido ≥ {ESTAB_MIN})")
    if f["estab_max"] > ESTAB_MAX:
        print(f"  - Estabilidad máxima: {f['estab_max']:.2f} cal (requerido ≤ {ESTAB_MAX})")
    if f["v_riel"] < V_RIEL_MIN:
        print(f"  - Velocidad de riel : {f['v_riel']:.1f} m/s (requerido ≥ {V_RIEL_MIN})"
              "  -> no depende de las aletas: motor, masa o largo del riel")
    if f["fs_flutter"] < FS_FLUTTER_MIN:
        print(f"  - FS flutter        : {f['fs_flutter']:.2f} (requerido ≥ {FS_FLUTTER_MIN})")


def tabla_resultados():
    tabla = pd.DataFrame(historial)
    tabla["error_apogeo"] = (tabla["apogeo"] - APOGEO_OBJETIVO).abs()
    return tabla.sort_values("puntaje")


def guardar_resultados(nombre_csv):
    if not historial:
        print("No se completó ninguna simulación.")
        return None
    tabla = tabla_resultados()
    tabla.drop(columns=["x"]).to_csv(nombre_csv, index=False)
    print(f"\nResultados guardados en {nombre_csv}")
    factibles = tabla[tabla["factible"]]
    print(f"Diseños factibles: {len(factibles)} de {len(tabla)}")

    if len(factibles) > 0:
        mejor = factibles.iloc[0]
        mostrar_diseno(mejor)
    else:
        print("Ningún diseño cumple todas las restricciones. Cuántas veces falló cada una:")
        print(f"  Estabilidad mínima : {(tabla['estab_min'] < ESTAB_MIN).sum()}")
        print(f"  Estabilidad máxima : {(tabla['estab_max'] > ESTAB_MAX).sum()}")
        print(f"  Velocidad de riel  : {(tabla['v_riel'] < V_RIEL_MIN).sum()}")
        print(f"  Flutter            : {(tabla['fs_flutter'] < FS_FLUTTER_MIN).sum()}")
        mejor = tabla.sort_values(["penalizacion", "error_apogeo"]).iloc[0]
        mostrar_diseno(mejor, "Mejor diseño disponible (NO cumple todas las restricciones)")
        mostrar_incumplimientos(mejor)

    if mejor["error_apogeo"] > 0.1 * APOGEO_OBJETIVO:
        print(f"\nAVISO: el apogeo queda a {mejor['error_apogeo']:,.0f} m del objetivo (> 10 %). "
              "Probablemente no es alcanzable cambiando solo las aletas: revisa motor o masa.")
    pegadas = variables_en_limite(mejor["x"])
    if pegadas:
        print("AVISO: estas variables quedaron en el límite de su rango: " + ", ".join(pegadas)
              + ". El óptimo podría estar fuera del rango permitido.")
    return mejor


def variables_en_limite(x):
    """Variables pegadas a un límite arbitrario (los bordes de flecha, posición y
    punta/raíz son diseños válidos: recto, al ras del fuselaje, rectangular)."""
    revisar = {0: "cuerda raíz", 2: "envergadura"}
    pegadas = []
    for i, nombre in revisar.items():
        lo, hi = LIMITES[i]
        margen = 0.01 * (hi - lo)
        if x[i] <= lo + margen or x[i] >= hi - margen:
            pegadas.append(nombre)
    lo, hi = LIMITES[4]
    fijo = (hi - lo) < 1e-5   # espesor fijo: no tiene sentido avisar
    if not fijo and (x[4] <= lo + 0.01 * (hi - lo) or x[4] >= hi - 0.01 * (hi - lo)):
        pegadas.append("espesor")
    return pegadas


# =============================================================================
# 7. MODOS DE EJECUCIÓN
# =============================================================================
def muestras_lhs(n):
    minimos = [lim[0] for lim in LIMITES]
    maximos = [lim[1] for lim in LIMITES]
    return qmc.scale(qmc.LatinHypercube(d=len(LIMITES), seed=42).random(n), minimos, maximos)


def explorar():
    print(f"\nExplorando {N_EXPLORAR} diseños...")
    for x in muestras_lhs(N_EXPLORAR):
        evaluar(x)


def optimizar(poblacion_inicial="latinhypercube"):
    print(f"\nOptimizando ({N_GENERACIONES} generaciones)...")
    res = differential_evolution(evaluar, LIMITES, maxiter=N_GENERACIONES, popsize=POBLACION_DE,
                                 seed=42, init=poblacion_inicial, polish=False, tol=1e-3)
    if not res.success:
        print("\nAVISO: la optimización llegó al límite de generaciones sin converger. "
              "Considera correrla con más generaciones.")

    print(f"\nRefinando el mejor diseño (hasta {ITER_REFINAMIENTO} simulaciones)...")
    minimize(evaluar, res.x, method="Nelder-Mead", bounds=LIMITES,
             options={"maxfev": ITER_REFINAMIENTO, "xatol": 1e-4, "fatol": 0.5})


def completo():
    explorar()
    n_poblacion = POBLACION_DE * len(LIMITES)
    mejores = tabla_resultados().head(n_poblacion)
    poblacion = np.array(mejores["x"].tolist())
    if len(poblacion) < n_poblacion:   # completar con muestras nuevas si faltan
        poblacion = np.vstack([poblacion, muestras_lhs(n_poblacion - len(poblacion))])
    print(f"\nIniciando la optimización desde los {len(mejores)} mejores diseños explorados.")
    optimizar(poblacion_inicial=poblacion)


def estimar_tiempo(modo):
    """Mide una simulación y estima la duración de la corrida."""
    x0 = np.array([(lo + hi) / 2 for lo, hi in LIMITES])
    inicio = time.time()
    simular(*convertir_variables(x0))
    seg = time.time() - inicio

    n_de = POBLACION_DE * len(LIMITES) * (N_GENERACIONES + 1) + ITER_REFINAMIENTO
    n = {"explorar": N_EXPLORAR, "optimizar": n_de, "completo": N_EXPLORAR + n_de}[modo]
    texto = "como máximo " if modo != "explorar" else ""
    print(f"\n≈ {seg:.2f} s por simulación → {texto}{n:,} simulaciones, unos {n * seg / 60:.0f} min.")


def guardar_ork(documento, archivo, f):
    base, extension = os.path.splitext(archivo)
    destino = os.path.abspath(f"{base}_finimized{extension or '.ork'}")
    simular(f["cr"], f["ct"], f["s"], f["sw"], f["t"], f["d"])

    Archivo = clase_java("java.io.File")
    Saver = clase_java("info.openrocket.core.file.GeneralRocketSaver",
                       "net.sf.openrocket.file.GeneralRocketSaver")
    errores = []
    if Saver is not None:
        saver = Saver()
        intentos = [lambda: saver.save(Archivo(destino), documento)]
        if hasattr(documento, "getDefaultStorageOptions"):
            intentos.append(lambda: saver.save(Archivo(destino), documento,
                                               documento.getDefaultStorageOptions()))
        for intento in intentos:
            try:
                intento()
                print(f"\nArchivo de OpenRocket guardado: {destino}")
                if not f["factible"]:
                    print("  Ojo: este diseño NO cumple todas las restricciones.")
                return
            except Exception as e:
                errores.append(str(e))
    print("\nNo se pudo guardar el .ork automáticamente.")
    for e in errores:
        print("  ", e)
    print("Ingresa las medidas del mejor diseño a mano en OpenRocket.")


PATRON_PRESET = re.compile(rb"\s*<preset\b[^>]*/>")


def quitar_presets(archivo):
    """
    OpenRocket sin interfaz gráfica no puede cargar referencias a la base de datos de
    componentes (presets). Crea una copia temporal del .ork sin esas referencias; las
    medidas de cada componente se conservan porque el .ork también las guarda.
    Devuelve (ruta a cargar, es_temporal).
    """
    if zipfile.is_zipfile(archivo):
        with zipfile.ZipFile(archivo) as z:
            entradas = [(info, z.read(info.filename)) for info in z.infolist()]
        n = sum(len(PATRON_PRESET.findall(d)) for i, d in entradas if i.filename.endswith(".ork"))
        if n == 0:
            return archivo, False
        fd, temporal = tempfile.mkstemp(suffix=".ork")
        os.close(fd)
        with zipfile.ZipFile(temporal, "w", zipfile.ZIP_DEFLATED) as z:
            for info, datos in entradas:
                if info.filename.endswith(".ork"):
                    datos = PATRON_PRESET.sub(b"", datos)
                z.writestr(info, datos)
    else:
        with open(archivo, "rb") as f:
            datos = f.read()
        n = len(PATRON_PRESET.findall(datos))
        if n == 0:
            return archivo, False
        fd, temporal = tempfile.mkstemp(suffix=".ork")
        with os.fdopen(fd, "wb") as f:
            f.write(PATRON_PRESET.sub(b"", datos))

    print(f"Se quitaron {n} referencias a la base de datos de componentes (presets) "
          "en una copia temporal; las medidas se conservan.")
    return temporal, True


# =============================================================================
# 8. PROGRAMA PRINCIPAL
# =============================================================================
def main():
    global orl, sim, aletas, LIMITES, LARGO_FUSELAJE, AXIAL_BOTTOM

    modos = ("explorar", "optimizar", "completo")
    if len(sys.argv) < 2 or sys.argv[1] not in modos:
        print("Uso: python finimizer.py [explorar | optimizar | completo] [archivo.ork]")
        return
    modo = sys.argv[1]
    archivo = sys.argv[2] if len(sys.argv) > 2 else ARCHIVO_ORK

    if not os.path.exists(archivo):
        print(f"No existe el archivo {archivo}")
        return
    archivo_carga, es_temporal = quitar_presets(archivo)
    try:
        ejecutar(modo, archivo, archivo_carga)
    finally:
        if es_temporal:
            os.remove(archivo_carga)


def ejecutar(modo, archivo, archivo_carga):
    global orl, sim, aletas, LIMITES, LARGO_FUSELAJE, AXIAL_BOTTOM

    with orlab.OpenRocketInstance() as instancia:
        orl = orlab.Helper(instancia)
        documento = orl.load_doc(archivo_carga)
        sim = documento.getSimulation(INDICE_SIMULACION)

        aletas = buscar_aletas(documento.getRocket())
        if aletas is None:
            print(f"No hay un set de aletas trapezoidales en {archivo}. Agrégalo en OpenRocket.")
            return
        fuselaje = aletas.getParent()
        if not hasattr(fuselaje, "getOuterRadius"):
            print("Las aletas no están en un fuselaje porta-aletas; no se puede calcular el diámetro.")
            return

        Axial = clase_java("info.openrocket.core.rocketcomponent.position.AxialMethod",
                           "net.sf.openrocket.rocketcomponent.position.AxialMethod")
        AXIAL_BOTTOM = Axial.BOTTOM if Axial is not None else None
        if AXIAL_BOTTOM is None:
            print("Aviso: no se pudo controlar la posición de las aletas en esta versión de OpenRocket.")

        diametro = 2 * fuselaje.getOuterRadius()
        LARGO_FUSELAJE = fuselaje.getLength()
        LIMITES = calcular_limites(diametro, LARGO_FUSELAJE)
        print(f"\nAletas encontradas: '{aletas.getName()}'")
        # La pestaña (fin tab) se toma tal cual del .ork; avisar si puede quedar más larga que la raíz
        try:
            largo_pestana = aletas.getTabLength()
            if largo_pestana > LIMITES[0][0]:
                print(f"Aviso: la pestaña mide {largo_pestana * 1000:.1f} mm y la cuerda raíz puede bajar "
                      f"hasta {LIMITES[0][0] * 1000:.1f} mm; revisa la pestaña en OpenRocket.")
        except Exception:
            pass
        print(f"Fuselaje porta-aletas: diámetro {diametro * 1000:.1f} mm, "
              f"largo {LARGO_FUSELAJE * 1000:.0f} mm")

        pedir_parametros(modo)
        estimar_tiempo(modo)
        if input("\n¿Continuar? [S/n]: ").strip().lower() == "n":
            return

        if modo == "explorar":
            explorar()
        elif modo == "optimizar":
            optimizar()
        else:
            completo()

        mejor = guardar_resultados(f"{modo}_aletas.csv")
        if mejor is not None:
            guardar_ork(documento, archivo, mejor)


if __name__ == "__main__":
    main()