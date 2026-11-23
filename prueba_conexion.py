#!/usr/bin/env python3
"""
prueba_conexion.py
Verifica que Python puede abrir tu .ork, encontrar las aletas, simular y modificarlas.

Uso:
    python prueba_conexion.py mi_cohete.ork
"""

import os
import re
import sys
import tempfile
import zipfile
import numpy as np
import orlab
from orlab import FlightDataType as T, FlightEvent as E

INDICE_SIMULACION = 0


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


def imprimir_arbol(componente, nivel=0):
    """Muestra todos los componentes del cohete con su tipo."""
    tipo = componente.getClass().getSimpleName()
    print("   " * nivel + f"- {componente.getName()}  [{tipo}]")
    for hijo in componente.getChildren():
        imprimir_arbol(hijo, nivel + 1)


def buscar_aletas(componente):
    """Devuelve el primer TrapezoidFinSet que encuentre."""
    if componente.getClass().getSimpleName() == "TrapezoidFinSet":
        return componente
    for hijo in componente.getChildren():
        encontrado = buscar_aletas(hijo)
        if encontrado is not None:
            return encontrado
    return None


def simular_y_resumir(orl, sim):
    orl.run_simulation(sim)
    datos = orl.get_timeseries(sim, [T.TYPE_TIME, T.TYPE_ALTITUDE,
                                     T.TYPE_VELOCITY_TOTAL, T.TYPE_STABILITY])
    tiempo = np.array(datos[T.TYPE_TIME], dtype=float)
    altitud = np.array(datos[T.TYPE_ALTITUDE], dtype=float)
    velocidad = np.array(datos[T.TYPE_VELOCITY_TOTAL], dtype=float)
    estabilidad = np.array(datos[T.TYPE_STABILITY], dtype=float)

    t_riel = float(orl.get_events(sim)[E.LAUNCHROD][0])
    validos = np.isfinite(estabilidad)
    return {
        "apogeo": np.nanmax(altitud),
        "v_max": np.nanmax(velocidad),
        "v_riel": np.interp(t_riel, tiempo, velocidad),
        "estab_riel": np.interp(t_riel, tiempo[validos], estabilidad[validos]),
    }


def imprimir_resumen(titulo, r):
    print(f"\n{titulo}")
    print(f"  Apogeo        : {r['apogeo']:.0f} m ({r['apogeo'] / 0.3048:,.0f} ft)")
    print(f"  V máxima      : {r['v_max']:.1f} m/s")
    print(f"  V salida riel : {r['v_riel']:.1f} m/s")
    print(f"  Estab. riel   : {r['estab_riel']:.2f} cal")


def main():
    if len(sys.argv) < 2:
        print("Uso: python prueba_conexion.py mi_cohete.ork")
        return
    archivo = sys.argv[1]
    if not os.path.exists(archivo):
        print(f"No existe el archivo {archivo}")
        return
    archivo_carga, es_temporal = quitar_presets(archivo)
    try:
        probar(archivo, archivo_carga)
    finally:
        if es_temporal:
            os.remove(archivo_carga)


def probar(archivo, archivo_carga):
    print("PASO 1: arrancando OpenRocket...")
    with orlab.OpenRocketInstance() as instancia:
        orl = orlab.Helper(instancia)
        print("  OK")

        print(f"\nPASO 2: cargando {archivo}")
        documento = orl.load_doc(archivo_carga)
        cohete = documento.getRocket()
        print("  Simulaciones en el archivo:")
        for i in range(documento.getSimulationCount()):
            print(f"    [{i}] {documento.getSimulation(i).getName()}")
        print("\n  Árbol de componentes:")
        imprimir_arbol(cohete)

        print("\nPASO 3: buscando aletas trapezoidales...")
        aletas = buscar_aletas(cohete)
        if aletas is None:
            print("  No hay un TrapezoidFinSet. El script de optimización solo maneja aletas trapezoidales.")
            return
        print(f"  Encontradas: '{aletas.getName()}'  <- usa este nombre en NOMBRE_ALETAS")
        print(f"  Cuerda raíz  : {aletas.getRootChord() * 1000:.1f} mm")
        print(f"  Cuerda punta : {aletas.getTipChord() * 1000:.1f} mm")
        print(f"  Envergadura  : {aletas.getHeight() * 1000:.1f} mm")
        print(f"  Flecha       : {aletas.getSweep() * 1000:.1f} mm")
        print(f"  Espesor      : {aletas.getThickness() * 1000:.2f} mm")
        print(f"  N° aletas    : {aletas.getFinCount()}")

        sim = documento.getSimulation(INDICE_SIMULACION)

        print("\nPASO 4: simulando con las aletas originales...")
        base = simular_y_resumir(orl, sim)
        imprimir_resumen("Resultado (compáralo con la interfaz de OpenRocket):", base)

        print("\nPASO 5: aumentando la envergadura 20 % para comprobar que el cambio se aplica...")
        envergadura_original = aletas.getHeight()
        aletas.setHeight(envergadura_original * 1.2)
        modificado = simular_y_resumir(orl, sim)
        aletas.setHeight(envergadura_original)
        imprimir_resumen("Resultado con aletas más grandes:", modificado)

        print("\nVERIFICACIÓN")
        if modificado["estab_riel"] > base["estab_riel"]:
            print("  OK  La estabilidad aumentó.")
        else:
            print("  ERROR  La estabilidad no aumentó: el cambio de geometría no está llegando a la simulación.")
        if modificado["apogeo"] < base["apogeo"]:
            print("  OK  El apogeo bajó (más arrastre y masa).")
        else:
            print("  AVISO  El apogeo no bajó: revisa si hay overrides de masa o de CG en el .ork.")


if __name__ == "__main__":
    main()
