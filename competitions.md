---
layout: page
title: Competitions
permalink: /competitions/
---

{% assign pubs = site.competitions | sort: 'year' | reverse %}
<ul>
{% for p in pubs %}
  <li>
    <a href="{{ p.url | relative_url }}">{{ p.title }}</a> ({{ p.year }})  
    {{ p.authors }} — <em>{{ p.venue }}</em>  
  </li>
{% endfor %}
</ul>


