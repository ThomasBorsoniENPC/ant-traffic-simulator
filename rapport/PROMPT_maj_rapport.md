# Prompt à passer au modèle qui maintient le rapport

> Dans le modèle d'agents de trafic bidirectionnel de fourmis, le terme de
> thigmotactisme (attraction de paroi) a été modifié, puis simplifié. Mets le
> rapport à jour en conséquence.
>
> **Ce qui a changé.** La loi de paroi n'est plus l'attraction pondérée par la
> composante tangentielle du cap. Elle est remplacée par un rappel vers une
> distance de consigne `d* = 1 mm` :
>
>     F_wall = c_wall · w(d) · sigma_wall
>     w(d) = (d − d*)/(l_max − d*)  si d ≥ d*        [attractif]
>     w(d) = (d − d*)/d*            si d < d*        [RÉPULSIF]
>
> avec `c_wall = 2` (au lieu de 10 : la loi précédente était fortement atténuée
> par sa pondération angulaire, la nouvelle applique `c_wall` plein à l'écart de
> consigne). `sigma_wall` est le vecteur unitaire de l'agent vers la paroi la
> plus proche.
>
> **Pourquoi.** L'ancienne loi était maximale quand l'agent longeait déjà la
> paroi, et n'était jamais répulsive : la paroi devenait un état absorbant. Un
> agent isolé, sans aucun voisin, y passait 98 % de son temps à moins d'1 mm du
> bord. Annuler la force près de la paroi ne suffisait pas (c'était déjà le cas
> sous 0,5 mm) : la zone morte était un fond de puits. Il fallait que la force
> change de signe.
>
> **Ce que ça produit, mesuré.** Une préférence de bord SANS écrasement : 3 %
> d'agents à moins de 0,5 mm de la paroi contre 30 % avec l'ancienne loi, plus
> de superposition entre agents, et le mélange transverse inchangé. Ce n'est PAS
> un équilibre stable : un agent isolé oscille encore sur toute la largeur du
> canal (distance à la paroi de 0,5 à 5 mm, moyenne 1,9 sur une demi-largeur de
> 5). Ne pas écrire « équilibre stable » ni « suivi de bord à distance
> constante » — écrire « préférence de bord ».
>
> **Une version intermédiaire à ne PAS documenter.** Une variante comportait un
> terme d'amortissement supplémentaire sur la vitesse normale
> (`− c_wall_damping · (e_theta · sigma_wall)`). Il a été retiré : la mesure a
> montré qu'il ne supprimait pas l'oscillation qu'il visait et ne déplaçait
> aucun observable. S'il apparaît quelque part dans le rapport, le supprimer.
>
> **Limite à conserver.** La distance de consigne vaut 1 mm dans le code, alors
> que les profils transverses mesurés sur les trajectoires réelles suggèrent
> 2,5 à 4 mm. Les longueurs n'ont pas été recalibrées ; c'est un point ouvert.
