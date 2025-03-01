# Laboratorio problema del coctel
## Descripción

En este laboratorio se abordó el problema del cóctel, que es como una fiesta en donde puede haber mucho ruido y múltiples personas hablando al mismo tiempo. Se trabajó con grabaciones obtenidas simulando este tipo de escenarios comunes, para así poder aislar y extraer la voz de un solo participante.
El hecho de extraer una voz en específico entre distintas señales puede ser un poco complicado, por lo que se hace uso del procesamiento digital de señales, de esta forma se puede enfocar en solo una fuente y asi filtrar el ruido y voces no deseadas. Además este documento muestra el paso a paso de como poder separar esta única voz, teniendo en cuenta también varias metodologías para la separación de señales, tales como ICA y BEamforming.


## Metodología experimental

### Organización
Para lograr obtener un escenario similar al de una fiesta tipo coctel, se simulo esta situación en un espacio silencioso y abierto. Primeramente se colocaron tres microfonos por todo el espacio y cada participante se ubicó en una posición adecuada entre los microfonos, después de esto se midio la distancia de cada microfono respecto a la persona, cómo se puede apreciar en la siguiente imágen se muestra como fue la distribución de todo por el espacio.

[![distribucion.jpg](https://i.postimg.cc/D0LKpXCL/distribucion.jpg)](https://postimg.cc/zHzMVVYG)

La medida de cada distancia desde la persona hasta cada micrófono es la siguiente

* Persona 1: Micro 1= 1,90 m, Micro 2= 4,90 m, Micro 3= 6,40 m
* Persona 2: Micro 1= 4,50 m, Micro 2= 2,40 m, Micro 3= 4,20 m
* Persona 3: Micro 1= 6,55 m, Micro 2= 4,50 m, Micro 3= 1,80 m

### Captar la señal

Después de que todas la personas estuvieran ubicadas en sus respectivas posiciones se inició la grabación en cada uno de los micrófonos, cada participante dijo  una frase diferente al mismo tiempo durante aproximadamente 15 segundos. Por último se hizo la grabación con todos los micrófonos del ruido ambiente, esto se hizo para poder hacer la relación señal-ruido.

## Procesamiento de señales

Cada una de las grabaciones se pasó a formato .wav, este es un formato compatible para el procesamiento de audio en Python. La libreria  soundfile es para leer y guardar archivos de audio en formato .wav y sklearn.decomposition es la libería que permite el Análisis de Componentes Independientes (ICA).

**Librerías**
```python
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sklearn.decomposition import FastICA
```

**Definición de parámetros**
Esta parte es fundamental, ya que con estos parámetros se establece el espacio de trabajo que usa beamforming para diseñar un arreglo para poder direccionar la señal deseada mientras se minimiza el ruido y las interferencias. Esto llega a facilitar el análisis y la separación de las fuentes de audio.

```python
#*Definir posiciones de los micrófonos en 2D (solo X, Y)*
mic_positions = np.array([[0.0, 0.0], [0.0, 6.0], [6.0, 6.0]]).T

#*Definir distancias de la Voz 3 a los micrófonos*
voice_3_distances = np.array([6.5, 4.5, 1.8])  # En metros

#*Velocidad del sonido en aire (m/s)*
speed_of_sound = 343

```

**Cargar grabaciones**

En esta parte se cargan las grabaciones de las voces y los ruidos
```python
#*Cargar grabaciones*
grabaciones = ["GRABACIÓN-1.wav", "Rinrin.wav", "Voz-micro-1.wav"]
ruidos = ["R3.wav", "SILENCIO-3.wav", "Ruido-Ambiente.wav"]

audio_signals = []
noise_signals = []
sample_rate = None
duraciones = []

for file in grabaciones:
    audio, sr = sf.read(file)
    if sample_rate is None:
        sample_rate = sr  # Guardamos la frecuencia de muestreo
    duraciones.append(len(audio))
    audio_signals.append(audio)

for file in ruidos:
    noise, sr = sf.read(file)
    noise_signals.append(noise[:min(duraciones)])  # Recortar a la duración mínima
```

Para que las señales queden del mismo tamaño se usa el siguiente código

```python
#*Recortar todas las señales a la misma duración mínima*
min_length = min(duraciones)
audio_signals = [audio[:min_length] for audio in audio_signals]
audio_matrix = np.column_stack(audio_signals)
noise_matrix = np.column_stack(noise_signals)
```

### Métodos de separación

**Aplicar beamforming**
El Beamforming Se trata de un sencillo proceso de medición de una sola toma, muy útil para elaborar mapas de presión sonora relativa y contribuciones de fuentes individuales a la intensidad sonora.El sistema hace una medición sencilla y elabora un mapa acústico de las fuentes de ruido, utilizando la matriz de micrófonos para detectar la dirección de llegada del sonido procedente de las fuentes. Para poder usar este método se hizo el siguiente código.

```python
#*Aplicar Beamforming basado en la Voz 3*
beamformed_signal = np.zeros_like(audio_matrix[:, 0])

for i in range(3):  # Para cada micrófono
    distancia = voice_3_distances[i]  # Distancia de la Voz 3 al micrófono
    delay = distancia / speed_of_sound  # Retardo en segundos
    sample_delay = int(delay * sample_rate)  # Convertir a muestras

    #*Alinear la señal con np.roll*
    beamformed_signal += np.roll(audio_matrix[:, i], -sample_delay)

beamformed_signal /= 3  # Promediar las señales alineadas
```

**Aplicación del ICA**
El análisis de componentes independientes (ICA) es una técnica que se utiliza para separar una señal multivariable en fuentes independientes no gaussianas. El ICA se puede utilizar para eliminar el ruido, extraer características y separar fuentes independientes de una señal mixta. En análisis de audio, cuando varias fuentes de sonido se capturan con múltiples micrófonos, las señales grabadas contienen una mezcla de todas las fuentes. Esta función descompone estas señales mixtas en componentes que sean estadísticamente independientes entre sí, lo que permite aislar diferentes voces o sonidos en un entorno ruidoso. El siguiente código permite usar este método para la separación de voces.

```python
#*Aplicar ICA para extraer la Voz 3*
ica = FastICA(n_components=3, max_iter=2000, tol=1e-6)
separated_sources = ica.fit_transform(np.column_stack([beamformed_signal, audio_matrix[:, 1], audio_matrix[:, 2]]))
```

Calculo SNR

```python
#*Cálculo del SNR*
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
# *Calcular SNR de cada grabación con cada ruido*
snr_values = np.zeros((3, 3))
for i in range(3):  # Para cada grabación
    for j in range(3):  # Para cada ruido
        snr_values[i, j] = calculate_snr(audio_signals[i], noise_signals[j])

#*Calcular SNR de la voz separada con cada ruido*
separated_snr_values = [calculate_snr(separated_sources[:, 0], noise_signals[j]) for j in range(3)]
```
Guardar la voz separada
```python
#*Guardar solo la voz separada de la Voz 3*
sf.write("voz_separada_3.wav", separated_sources[:, 0], sample_rate)
```
Mostrar los datos en pantalla, tanto de los SNR como de que si se guardo el archivo de la voz separada
```python
#*Mostrar resultados de SNR*
for i in range(3):
    for j in range(3):
        print(f"SNR de la Grabación {i+1} con Ruido {j+1}: {snr_values[i, j]:.2f} dB")

for j in range(3):
    print(f"SNR de la Voz Separada con Ruido {j+1}: {separated_snr_values[j]:.2f} dB")

print("Voz 3 separada correctamente y guardada en 'voz_separada_3.wav'.")
```
[![SNR.jpg](https://i.postimg.cc/4dFCx10z/SNR.jpg)](https://postimg.cc/34pfL2YR)

Para analizar los resultados obtenidos de SNR, se deben ver primero las señales originales, en donde se evidencia una gran afectación por el ruido, especialmente en la grabación 3, que presenta incluso valores negativos. La grabación 1 tiene los mejores SNR entre las grabaciones originales, que van de 2.45dB a 8.81dB, mientras que las otras  dos presentaron valores más bajos, llegando hasta -6.30dB, lo que indica que el ruido es más fuerte que la señal. Así mismo se observa que el ruido 2 es más problemático, ya que en casi todas las grabaciones originales presenta los SNR más bajos, lo que dificulta la separación de la señal.
Para el caso de la voz separada, se ve una mejoría notoria en los SNR, alcanzando valores de 25.07dB a 33.34dB, lo que demuestra que  el método de separación de voz ha sido altamente efectivo en la eliminación de ruido. Se observa que el ruido 3 es el más fácil de filtrar, ya que el SNR más alto corresponde a la voz separada con este ruido. En general, la técnica utilizada logró aumentar el SNR a más de 20dB, lo que confirma que el proceso de separación fue exitoso.
## Análisis temporal y espectral

En esta parte se muestra la señal de audio en el dominio del tiempo y de la frecuencia usando la Transformada Rápida de Fourier (FFT) y la Transformada Discreta de Fourier (DFT). Esto se hace para visualizar cómo varía la señal en el tiempo y su distribución espectral, lo que permite entender como se compone la voz y su posible separación de otras fuentes de ruido.

```python
# Análisis temporal y espectral
def plot_signal_and_spectrum(signal, sr, title, color):
    plt.figure(figsize=(12, 8))
    
    #Dominio del tiempo
    plt.subplot(3, 1, 1)
    time_axis = np.linspace(0, len(signal) / sr, num=len(signal))
    plt.plot(time_axis, signal, color=color)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.title(f"Señal en el tiempo - {title}")
    
    #Espectro de frecuencia con FFT
    plt.subplot(3, 1, 2)
    freq_axis = np.fft.rfftfreq(len(signal), d=1/sr)
    spectrum = np.abs(fft(signal)[:len(freq_axis)])
    plt.plot(freq_axis[freq_axis <= 500], spectrum[freq_axis <= 500], color=color)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.title(f"Espectro de Frecuencia (FFT) - {title}")
    
    #Espectro de frecuencia con DFT
    plt.subplot(3, 1, 3)
    dft_freq_axis = np.fft.rfftfreq(len(signal), d=1/sr)
    dft_spectrum = np.abs(fft(signal)[:len(dft_freq_axis)])
    plt.plot(dft_freq_axis[dft_freq_axis <= 500], dft_spectrum[dft_freq_axis <= 500], color=color)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.title(f"Espectro de Frecuencia (DFT) - {title}")
    
    plt.tight_layout()
    plt.show()

#Aplicar análisis a cada señal de micrófono
colors = ['r', 'g', 'b']
for i, audio in enumerate(audio_signals):
    plot_signal_and_spectrum(audio, sample_rate, f"Grabacion {i+1}", colors[i])

#Aplicar análisis a la voz separada
plot_signal_and_spectrum(separated_sources[:, 0], sample_rate,"voz separada",'m')
```
[![g1.jpg](https://i.postimg.cc/qBPKtHy5/g1.jpg)](https://postimg.cc/NLxFVVN6)
[![g2.jpg](https://i.postimg.cc/BngPw8yr/g2.jpg)](https://postimg.cc/yDDNDN7j)
[![g3.jpg](https://i.postimg.cc/Dzw4Vv38/g3.jpg)](https://postimg.cc/MngpV8TS)
[![voz.jpg](https://i.postimg.cc/DZWWZxZh/voz.jpg)](https://postimg.cc/k6mMYFKY)

## Análisis
En todas las cuatro graficas extraídas a partir de las voces de los audios capturados y del audio de voz separada tenemos lo siguiente:

La primera gráfica representa la señal en el dominio del tiempo, mostrando cómo varía la amplitud a lo largo de los segundos. Se observa que la señal tiene cambios notorios en su intensidad, lo cual es característico de la voz humana, con momentos de mayor y menor energía. Estas variaciones pueden deberse a pausas naturales en el habla, cambios en la entonación o la presencia de ruido ambiental. Además, la amplitud no es constante, lo que indica que la señal contiene información dinámica propia del habla y posibles interferencias.

La segunda gráfica muestra el espectro de frecuencia obtenido mediante la Transformada Rápida de Fourier (FFT). Aquí se observa cómo la energía de la señal se distribuye a través de diferentes frecuencias. Se destacan picos en las frecuencias bajas y medias, especialmente alrededor de los 100 Hz, lo cual es característico de la voz humana, ya que la frecuencia fundamental de la mayoría de las voces adultas se encuentra en ese rango. También se pueden notar componentes en frecuencias más altas, que pueden deberse a armónicos naturales del habla o a ruido presente en la grabación.

La tercera gráfica corresponde al espectro obtenido con la Transformada Discreta de Fourier (DFT). Su distribución espectral es muy similar a la obtenida con la FFT, ya que ambas transformadas describen la misma información de frecuencia. En este caso la FFT es mas adecuada para el análisis de señales de audio, ya que la composición de una señal de audio puede contener un gran numero de muestras por segundo. Es por eso que en términos de eficiencia, la FFT es mejor para esta aplicación, a pesar de que la implementación de la DFT nos brinde el mismo resultado.

## Conclusiones

Se pudo concluir que; gracias al análisis hecho en las señales de audio capturadas y separadas, la Transformada Rápida de Fourier (FFT) y la Transformada Discreta de Fourier (DFT) tienen distribuciones similares de energía en el espectro de frecuencias, destacando componentes principales alrededor de los 100 Hz, típicos de la voz humana. Sin embargo, la FFT demostró ser más eficiente en el procesamiento de señales de audio debido a su mayor efectividad de procesamiento de señales y muestras por segundo.  

Por parte del análisis de la relación señal-ruido (SNR), se evidenció una fuerte afectación por ruido en las grabaciones originales, con valores negativos en algunos casos, indicando que la interferencia era superior a la señal útil. Tras la separación de voz, los SNR mejoraron significativamente, alcanzando valores superiores a 25 dB, lo que confirma la efectividad del método utilizado. No obstante, los resultados no fueron óptimos debido a las condiciones de grabación, ya que el medio y los micrófonos de grabación no permitieron grabar de la manera esperada las señales empleadas en esta practica. 

## Bibliografía

Beamforming e Holografia acústica | Brüel & Kjær. (n.d.). Brüel & Kjær | B&K | Sound and Vibration. https://www.bksv.com/es/knowledge/applications/noise-source-identification/beamforming
Amir. (2024, June 15). Independent Component Analysis (ICA) with python code. Medium. https://medium.com/@ab.jannatpour/independent-component-analysis-ica-with-python-code-e7d1dd290241


### Colaboradores
* Youling Andrea Orjuela Bermúdez (5600815)
* José Manuel Gómez Carrillo (5600793)
* Juan Camilo Quintero Velandia (5600745)
