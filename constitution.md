# constitution.md

Constitución del proyecto de detección de fraude. Define los principios, reglas y restricciones fundamentales que todo agente y toda tarea debe respetar durante el desarrollo, sin excepción.

La separación de responsabilidades es:

- `agents.md` → cómo trabajan los agentes.
- `constitution.md` → qué principios y reglas deben respetarse siempre.
- `specs.md` → qué debe hacer concretamente el sistema.

## 1. Integridad de los datos

- Debe preservarse siempre el dataset raw, sin modificaciones in-place.
- Toda transformación debe ser trazable y reproducible.
- Queda prohibido ocultar problemas de calidad mediante transformaciones silenciosas.

## 2. Reproducibilidad

- Los resultados deben poder reproducirse a partir del código, los datos y el registro de ejecución.
- Deben quedar registrados: transformaciones, splits, configuraciones y experimentos relevantes.

## 3. Data leakage

- Regla fundamental: ninguna información que no estaría disponible en el momento de inferencia puede utilizarse para construir una feature ni para entrenar el modelo.
- Debe prestarse especial atención a la información temporal y a las agregaciones históricas.

## 4. Feature engineering

- Toda feature debe tener una justificación explícita.
- Debe distinguirse entre features disponibles en producción y features que dependen de información futura.
- Queda prohibido crear features únicamente para mejorar artificialmente una métrica.

## 5. Evaluación

- La evaluación debe representar razonablemente el escenario real de detección de fraude.
- Queda prohibido depender exclusivamente de accuracy.
- El desbalance de clases debe considerarse explícitamente en la elección de métricas y procedimiento de evaluación.
- El conjunto de test debe mantenerse independiente de la selección de features y del tuning de hiperparámetros.

## 6. Transparencia metodológica

- Las decisiones importantes deben estar justificadas.
- Los resultados negativos o las decisiones que no mejoren el modelo no deben ocultarse.
- Queda prohibido manipular la metodología para obtener mejores resultados.

## 7. Simplicidad y mantenibilidad

- Debe preferirse la solución más simple que resuelva correctamente el problema.
- No debe introducirse una herramienta, modelo o infraestructura sin una razón concreta.
- Debe mantenerse una separación clara de responsabilidades.

## 8. Producción e inferencia

- Las transformaciones utilizadas durante el entrenamiento deben poder reproducirse de manera consistente durante la inferencia.
- Queda prohibido utilizar información que solo exista durante el entrenamiento si no puede obtenerse en producción.

## 9. Trazabilidad de experimentos

- Los experimentos importantes deben poder relacionarse con el código, los datos, la configuración y los resultados que los produjeron.

## 10. Honestidad de los resultados

- No deben presentarse métricas de validación o test como evidencia de rendimiento real sin aclarar sus limitaciones.
- Debe diferenciarse claramente entre resultados experimentales y comportamiento esperado en producción.

## Estatus del documento

- Ninguno de estos principios puede violarse en favor de conveniencia, velocidad o resultados aparentemente mejores.
- Cualquier modificación a esta constitución requiere una decisión explícita, justificada y registrada.
- Los detalles de implementación (carpetas, arquitectura, herramientas, métricas concretas) pertenecen a `specs.md`.