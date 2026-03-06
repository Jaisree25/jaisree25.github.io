---
layout: page
title: Certifications
permalink: /certifications/
---

<div class="card-grid">
{% assign items = site.certifications | sort: 'date' | reverse %}
{% for p in items %}
  <a class="card" href="{{ p.url | relative_url }}">
    <h3>{{ p.title }}</h3>
    {% if p.date %}<div class="meta">{{ p.date | date: "%B %Y" }}</div>{% endif %}
    {% if p.summary %}<p>{{ p.summary }}</p>{% endif %}
  </a>
{% endfor %}
</div>
