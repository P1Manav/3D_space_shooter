# Create a .drawio XML structure for the PPO workflow in the 3D Space Shooter game

drawio_xml = """<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net">
  <diagram id="PPO_Workflow" name="PPO_Workflow">
    <mxGraphModel dx="2323" dy="1611" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1"
    fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
    <root>
        <!-- Default Layer -->
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>

        <!-- Title Label -->
        <mxCell id="title" value="PPO for 3D Space Shooter" style="text;html=1;align=center;verticalAlign=middle;fontSize=18;fontStyle=1;fontColor=#FFFFFF;fillColor=#00897B;strokeColor=none;rounded=1;spacing=8;" vertex="1" parent="1">
        <mxGeometry x="300" y="20" width="300" height="50" as="geometry"/>
        </mxCell>

        <!-- Environment Box -->
        <mxCell id="envBox" value="1) Environment (3D Space Shooter)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#BCAAA4;strokeWidth=2;fontSize=12;" vertex="1" parent="1">
        <mxGeometry x="40" y="100" width="200" height="80" as="geometry"/>
        </mxCell>
        <mxCell id="envText" value="State: (Ship Pos, Enemy Pos, Bullets, etc.)" style="text;html=1;align=center;verticalAlign=middle;fontSize=10;strokeColor=none;fillColor=none;" vertex="1" parent="envBox">
        <mxGeometry x="10" y="10" width="180" height="60" as="geometry"/>
        </mxCell>

        <!-- Agent (Policy) Box -->
        <mxCell id="agentBox" value="2) Agent (Policy Network)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=2;fontSize=12;" vertex="1" parent="1">
        <mxGeometry x="320" y="100" width="200" height="80" as="geometry"/>
        </mxCell>
        <mxCell id="agentText" value="Outputs Continuous Actions (Thrust, Turn, Fire, etc.)" style="text;html=1;align=center;verticalAlign=middle;fontSize=10;strokeColor=none;fillColor=none;" vertex="1" parent="agentBox">
        <mxGeometry x="10" y="10" width="180" height="60" as="geometry"/>
        </mxCell>

        <!-- Memory/Buffer Box -->
        <mxCell id="memoryBox" value="3) Memory / Buffer" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=2;fontSize=12;" vertex="1" parent="1">
        <mxGeometry x="600" y="100" width="240" height="80" as="geometry"/>
        </mxCell>
        <mxCell id="memoryText" value="Stores (s, a, r, s'), log probs, values..." style="text;html=1;align=center;verticalAlign=middle;fontSize=10;strokeColor=none;fillColor=none;" vertex="1" parent="memoryBox">
        <mxGeometry x="10" y="10" width="220" height="60" as="geometry"/>
        </mxCell>

        <!-- PPO Optimization Box -->
        <mxCell id="ppoBox" value="4) PPO Optimization" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#66BB6A;strokeWidth=2;fontSize=12;" vertex="1" parent="1">
        <mxGeometry x="300" y="240" width="300" height="120" as="geometry"/>
        </mxCell>
        <mxCell id="ppoText" value="Compute Advantages (GAE)&#xa;Clipped Objective&#xa;Value Loss + Entropy Bonus&#xa;Update Policy" style="text;html=1;align=center;verticalAlign=middle;fontSize=10;strokeColor=none;fillColor=none;" vertex="1" parent="ppoBox">
        <mxGeometry x="10" y="10" width="280" height="100" as="geometry"/>
        </mxCell>

        <!-- Arrows for State/Action/Reward Flow -->
        <!-- State from Env to Agent -->
        <mxCell id="arrowEnvToAgent" style="endArrow=block;endFill=1;strokeWidth=2;strokeColor=#424242;" edge="1" parent="1">
        <mxGeometry relative="1" as="geometry">
            <mxPoint x="0" y="0" as="sourcePoint"/>
            <mxPoint x="0" y="0" as="targetPoint"/>
        </mxGeometry>
        <mxCell id="envToAgentLbl" value="State (S_t)" style="text;html=1;align=center;verticalAlign=middle;fontSize=10;strokeColor=none;fillColor=none;labelBackgroundColor=none;" vertex="1" connectable="0" parent="arrowEnvToAgent">
            <mxGeometry x="0.5" y="0.5" width="80" height="20" relative="1" as="geometry">
            <mxGeometry as="offset" x="-40" y="-20"/>
            </mxGeometry>
        </mxCell>
        <mxCell id="envToAgentConnector" edge="1" connectable="0" parent="arrowEnvToAgent" source="envBox" target="agentBox">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        </mxCell>

        <!-- Action from Agent to Env -->
        <mxCell id="arrowAgentToEnv" style="endArrow=block;endFill=1;strokeWidth=2;strokeColor=#424242;" edge="1" parent="1">
        <mxGeometry relative="1" as="geometry">
            <mxPoint x="0" y="0" as="sourcePoint"/>
            <mxPoint x="0" y="0" as="targetPoint"/>
        </mxGeometry>
        <mxCell id="agentToEnvLbl" value="Action (A_t)" style="text;html=1;align=center;verticalAlign=middle;fontSize=10;strokeColor=none;fillColor=none;" vertex="1" connectable="0" parent="arrowAgentToEnv">
            <mxGeometry x="0.5" y="0.5" width="80" height="20" relative="1" as="geometry">
            <mxGeometry as="offset" x="-40" y="-20"/>
            </mxGeometry>
        </mxCell>
        <mxCell id="agentToEnvConnector" edge="1" connectable="0" parent="arrowAgentToEnv" source="agentBox" target="envBox">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        </mxCell>

        <!-- Reward from Env to Agent -->
        <mxCell id="arrowEnvRewardToAgent" style="endArrow=block;endFill=1;strokeWidth=2;strokeColor=#424242;" edge="1" parent="1">
        <mxGeometry relative="1" as="geometry">
            <mxPoint x="0" y="0" as="sourcePoint"/>
            <mxPoint x="0" y="0" as="targetPoint"/>
        </mxGeometry>
        <mxCell id="envRewardToAgentLbl" value="Reward (R_t)" style="text;html=1;align=center;verticalAlign=middle;fontSize=10;strokeColor=none;fillColor=none;" vertex="1" connectable="0" parent="arrowEnvRewardToAgent">
            <mxGeometry x="0.5" y="0.5" width="80" height="20" relative="1" as="geometry">
            <mxGeometry as="offset" x="-40" y="-20"/>
            </mxGeometry>
        </mxCell>
        <mxCell id="envRewardToAgentConnector" edge="1" connectable="0" parent="arrowEnvRewardToAgent" source="envBox" target="agentBox">
            <mxGeometry relative="1" as="geometry" x="0.5" y="1">
            <mxPoint as="offset" x="0" y="30"/>
            </mxGeometry>
        </mxCell>
        </mxCell>

        <!-- Arrow from Agent to Memory -->
        <mxCell id="arrowAgentToMem" style="endArrow=block;endFill=1;strokeWidth=2;strokeColor=#424242;" edge="1" parent="1">
        <mxGeometry relative="1" as="geometry">
            <mxPoint x="0" y="0" as="sourcePoint"/>
            <mxPoint x="0" y="0" as="targetPoint"/>
        </mxGeometry>
        <mxCell id="agentToMemLbl" value="(s, a, r, s')" style="text;html=1;align=center;verticalAlign=middle;fontSize=10;strokeColor=none;fillColor=none;labelBackgroundColor=none;" vertex="1" connectable="0" parent="arrowAgentToMem">
            <mxGeometry x="0.5" y="0.5" width="80" height="20" relative="1" as="geometry">
            <mxGeometry as="offset" x="-40" y="-20"/>
            </mxGeometry>
        </mxCell>
        <mxCell id="agentToMemConnector" edge="1" connectable="0" parent="arrowAgentToMem" source="agentBox" target="memoryBox">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        </mxCell>

        <!-- Arrow from Memory to PPO -->
        <mxCell id="arrowMemToPPO" style="endArrow=block;endFill=1;strokeWidth=2;strokeColor=#424242;" edge="1" parent="1">
        <mxGeometry relative="1" as="geometry">
            <mxPoint x="0" y="0" as="sourcePoint"/>
            <mxPoint x="0" y="0" as="targetPoint"/>
        </mxGeometry>
        <mxCell id="memToPPOLbl" value="Rollouts / Trajectories" style="text;html=1;align=center;verticalAlign=middle;fontSize=10;strokeColor=none;fillColor=none;labelBackgroundColor=none;" vertex="1" connectable="0" parent="arrowMemToPPO">
            <mxGeometry x="0.5" y="0.5" width="100" height="20" relative="1" as="geometry">
            <mxGeometry as="offset" x="-50" y="-20"/>
            </mxGeometry>
        </mxCell>
        <mxCell id="memToPPOConnector" edge="1" connectable="0" parent="arrowMemToPPO" source="memoryBox" target="ppoBox">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        </mxCell>

        <!-- Arrow from PPO to Agent (Policy Update) -->
        <mxCell id="arrowPPOToAgent" style="endArrow=block;endFill=1;strokeWidth=2;strokeColor=#424242;" edge="1" parent="1">
        <mxGeometry relative="1" as="geometry">
            <mxPoint x="0" y="0" as="sourcePoint"/>
            <mxPoint x="0" y="0" as="targetPoint"/>
        </mxGeometry>
        <mxCell id="ppoToAgentLbl" value="Updated Policy (θ_new)" style="text;html=1;align=center;verticalAlign=middle;fontSize=10;strokeColor=none;fillColor=none;labelBackgroundColor=none;" vertex="1" connectable="0" parent="arrowPPOToAgent">
            <mxGeometry x="0.5" y="0.5" width="100" height="20" relative="1" as="geometry">
            <mxGeometry as="offset" x="-50" y="-20"/>
            </mxGeometry>
        </mxCell>
        <mxCell id="ppoToAgentConnector" edge="1" connectable="0" parent="arrowPPOToAgent" source="ppoBox" target="agentBox">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        </mxCell>

    </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

# Save as .drawio file
file_path = "PPO_3D_Space_Shooter.drawio"
with open(file_path, "w", encoding="utf-8") as file:
    file.write(drawio_xml)


file_path
