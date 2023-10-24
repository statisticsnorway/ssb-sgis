Network analysis
================

This page includes all public network analysis functionality of sgis.

The actual analysis is done in the NetworkAnalysis class.

Rules for the analyses is set with the NetworkAnalysisRules class.

The GeoDataFrame of line geometries can be prepared for analysis with the functions
in the remaining sections. The 'Nodes' section is used internally in other functions,
and is not commonly needed.

.. toctree::
   :maxdepth: 3

   networkanalysis
   networkanalysisrules
   directednetwork
   finding_isolated_networks
   closing_network_holes
   cutting_lines
   traveling_salesman
   nodes
