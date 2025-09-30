---
layout: page
title: Certifications
permalink: /certifications/
---

<div class="card-grid">
{% assign items = site.certifications | sort: 'year' | reverse %}
{% for p in items %}
  <a class="card" href="{{ p.url | relative_url }}">
    <h3>{{ p.title }}</h3>
    {% if p.year %}<div class="meta">{{ p.year }}</div>{% endif %}
    {% if p.summary %}<p>{{ p.summary }}</p>{% endif %}
  </a>
{% endfor %}
</div>
