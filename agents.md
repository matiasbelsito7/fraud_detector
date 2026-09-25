# agents.md

Guía de trabajo para agentes de IA que intervengan en este repositorio.

## 1. Contexto mínimo

- Proyecto de Machine Learning para detección de fraude.
- Dataset base: IEEE-CIS Fraud Detection.
- Desarrollo incremental orientado a ML Engineering/MLOps.
- `agents.md` define el comportamiento y flujo de trabajo de los agentes. No reemplaza a `constitution.md` ni a `specs.md`.

## 2. Flujo de trabajo de los agentes

1. Inspeccionar el repositorio y el estado actual del proyecto antes de modificar cualquier archivo.
2. Comprender el contexto: leer configuración, documentación y dependencias existentes.
3. Planificar cambios importantes y exponer el plan antes de implementarlo.
4. Trabajar de manera incremental, en pasos pequeños y verificables.
5. Verificar los cambios realizados después de cada paso.
6. Documentar las decisiones relevantes y sus motivos.
7. No asumir que una funcionalidad existe: comprobarlo en el código antes de usarla.

## 3. Responsabilidades

### Al analizar código existente
- Entender qué hace el código y por qué, antes de proponer o hacer cambios.
- Describir el comportamiento encontrado sin modificar archivos salvo que se pida.

### Al implementar una nueva funcionalidad
- Seguir los patrones y convenciones ya presentes en el proyecto.
- Implementar solo lo necesario para la tarea.
- Verificar que la funcionalidad funcione antes de darla por terminada.

### Al modificar una funcionalidad existente
- Evaluar el impacto del cambio en el resto del sistema.
- Mantener el cambio acotado y reversible.
- No romper comportamiento existente sin justificación explícita.

### Al encontrar un problema
- Reproducir el problema y aislar la causa antes de corregirlo.
- Reportar el problema con contexto suficiente; corregir solo si es seguro y está dentro del alcance.

### Ante una decisión de arquitectura
- Consultar `constitution.md` y `specs.md` antes de decidir.
- Proponer opciones con trade-offs y recomendar una.
- Registrar la decisión y su justificación.

### Al agregar o modificar tests
- Alinearse con el framework de tests existente; si no existe, proponer uno antes de inventar uno.
- Actualizar los tests afectados junto con el código modificado.
- Verificar que los tests pasen antes de finalizar.

### Al terminar una tarea
- Verificar que todo lo implementado funcione correctamente.
- Resumir los cambios realizados y las decisiones tomadas.
- Indicar trabajo pendiente o riesgos conocidos.

## 4. Interacción con otros documentos

- `constitution.md`: reglas, principios y restricciones innegociables del proyecto.
- `specs.md`: requisitos funcionales, técnicos y comportamiento esperado del sistema.
- Consultar y respetar ambos documentos cuando correspondan, además de la documentación adicional del repositorio.
- Si `constitution.md` o `specs.md` no existen aún, no inventar sus reglas: señalarlo cuando una tarea las requiera.

## 5. Buenas prácticas

- Mantener los cambios acotados a la tarea asignada.
- Evitar modificaciones innecesarias fuera de alcance.
- Mantener la separación de responsabilidades.
- Evitar duplicación de lógica.
- Priorizar código comprensible y mantenible.
- No introducir complejidad sin una razón concreta.
- No agregar comentarios salvo que aporten valor real o se soliciten.

## 6. Protocolo ante ambigüedades

- **Faltan requisitos:** preguntar o consultar la documentación antes de asumir; nunca inventar requisitos.
- **Varias soluciones posibles:** compararlas usando los criterios del proyecto, recomendar una y justificar la elección.
- **Contradicción detectada:** (p. ej. entre documentos o entre documentación y código) reportarla explícitamente y no resolverla unilateralmente.
- **Problema que afecta decisiones anteriores:** informar antes de continuar; revisar si las decisiones previas deban revertirse o ajustarse.