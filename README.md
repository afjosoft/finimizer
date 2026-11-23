# Finimizer

Optimizador de aletas trapezoidales para cohetes, usando [OpenRocket](https://openrocket.info) como motor de simulación a través de [orlab](https://pypi.org/project/orlab/).

Dado un cohete modelado en OpenRocket, Finimizer busca la geometría de aletas que deja el apogeo lo más cerca posible del objetivo, cumpliendo restricciones de estabilidad, velocidad de salida del riel y flutter. Los valores por defecto siguen la IREC DTEG (Design, Test & Evaluation Guide).

## Requisitos

- Python 3.10 o superior
- JDK 17 o 21
- OpenRocket 24.12 o 23.09 (orlab descarga el jar)

## Instalación

```bash
git clone <url-del-repo>
cd finimizer
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m orlab fetch
```

## Preparar el archivo de OpenRocket

1. El cohete debe tener un set de **aletas trapezoidales** dentro de un fuselaje (body tube).
2. Configura en OpenRocket lo que Finimizer **no** optimiza: material de las aletas, perfil del borde, filetes, pestaña, motor, riel y condiciones de lanzamiento.
3. Quita los *overrides* de masa y CG del cohete y de las aletas; si no, los cambios de geometría no afectan la masa.
4. Verifica la conexión:

```bash
python prueba_conexion.py mi_cohete.ork
```

## Uso

```bash
python finimizer.py explorar  mi_cohete.ork   # muestreo del espacio de diseño
python finimizer.py optimizar mi_cohete.ork   # evolución diferencial + refinamiento
python finimizer.py completo  mi_cohete.ork   # explora y optimiza desde los mejores (recomendado)
```

| Modo | Para qué sirve |
|---|---|
| `explorar` | Ver si hay diseños factibles y qué restricción manda |
| `optimizar` | Encontrar el diseño óptimo |
| `completo` | Diseño final: explora y luego optimiza desde los mejores diseños |

El programa pide los parámetros de diseño (Enter = valor por defecto):

- Apogeo objetivo [m]
- Estabilidad mínima y máxima durante el vuelo [cal]
- Velocidad mínima de salida del riel [m/s]
- Factor de seguridad mínimo contra flutter
- Espesor mínimo y máximo de las aletas [mm] (obligatorios)
- Cuerda punta mínima [mm] y ángulo máximo de flecha [°] (opcionales)
- Material de las aletas (define el módulo de cortante G para el flutter)
- Número de aletas (3 o 4)

## Variables de diseño

Cuerda raíz, cuerda punta, envergadura, flecha, espesor y posición de las aletas a lo largo del fuselaje. Toda combinación es geométricamente válida: la punta nunca sobresale del final del fuselaje.

## Resultados

- `<modo>_aletas.csv`: todos los diseños evaluados.
- `<archivo>_finimized.ork`: copia del cohete con el mejor diseño aplicado.
- En pantalla: el mejor diseño, o el mejor disponible con las restricciones que incumple, y avisos si el apogeo no es alcanzable o si alguna variable quedó en el límite de su rango.

## Valores por defecto (IREC DTEG)

| Parámetro | Valor | Sección |
|---|---|---|
| Estabilidad mínima | 1.5 cal | 10.3.1 |
| Estabilidad máxima | 4.0 cal (estática) / 6.0 (dinámica) | 10.4.1 |
| Velocidad de salida del riel | 30 m/s (mínimo absoluto 15 m/s) | 10.2 |
| Factor de seguridad de flutter | 1.5 | 8.2.2 |

Tomados de la DTEG rev. 2024 v1.1.3 (IREC 2025). Verificar contra la versión vigente.

## Limitaciones

- **Flutter:** se estima con la fórmula de NACA TN 4197 (forma de Apogee *Peak of Flight* #291). Supone una placa plana de material isotrópico y no considera la flecha. Sirve como filtro de diseño; el diseño final debe verificarse con un método más completo y, si es posible, con ensayos.
- **Módulo de cortante G:** los valores del menú son aproximados. Usar datos medidos del laminado real.
- **Aerodinámica:** la de OpenRocket es aproximada en régimen transónico y supersónico; validar con otra herramienta si el cohete supera Mach ~0.9.
- **Componentes del catálogo:** OpenRocket sin interfaz gráfica no carga referencias a la base de datos de componentes (*presets*). Finimizer las quita en una copia temporal; las medidas se conservan.