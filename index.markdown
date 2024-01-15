---
layout: home
title: null
---

`HELLO! The site is still under construction :  ( You may want to visit my resume first :  )`


`16 Jan 2023: Updated second note.`

`7 Jan 2023: Updated first note.`

`26 Dec 2023: First commit. template ready, resume ready.`




`Recently Updated:`
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>

