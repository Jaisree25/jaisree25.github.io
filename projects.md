---
layout: page
title: Projects
permalink: /projects/
---

{% assign items = site.projects | sort: 'year' | reverse %}
<ul>
{% for p in items %}
  <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a>{% if p.year %} — {{ p.year }}{% endif %}</li>
{% endfor %}
</ul>
