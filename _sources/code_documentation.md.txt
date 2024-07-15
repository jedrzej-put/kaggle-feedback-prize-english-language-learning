# Code documentation

```{hint}

  To add your code use sphinx tool in project root directory:

    $ sphinx-apidoc -o docs/api/ src/feedback_prize_english_language_learning

  and add reference from any page which is reachable from the index page.
```

```python
    import feedback_prize_english_language_learning
```

```{toctree}
---
maxdepth: 4
---
api/modules
```