---
layout: page
title: Competitions
permalink: /competitions/
---

{% assign pubs = site.competitions | sort: 'year' | reverse %}
<ul style="line-height: 2;">
{% for p in pubs %}
  <li style="margin-bottom: 0.6rem;">
    <a href="{{ p.url | relative_url }}">{{ p.title }}</a> ({{ p.year }})  
    {{ p.authors }} — <em>{{ p.venue }}</em>  
  </li>
{% endfor %}
</ul>


