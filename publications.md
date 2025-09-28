---
layout: page
title: Publications
permalink: /publications/
published: false
---

{% assign pubs = site.publications | sort: 'year' | reverse %}
<ul>
{% for p in pubs %}
  <li>
    <strong>{{ p.title }}</strong> ({{ p.year }})  
    {{ p.authors }} — <em>{{ p.venue }}</em>  
    {% if p.paper_url %}[PDF]({{ p.paper_url }}){% endif %}
    {% if p.code_url %} • [Code]({{ p.code_url }}){% endif %}
  </li>
{% endfor %}
</ul>
