import click
import cv2

import stretch_mujoco
from stretch_mujoco.enums.stretch_cameras import StretchCameras


@click.command()
@click.option(
    "--scene-xml-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a scene XML file.",
)
@click.option("--headless", is_flag=True, help="Run the simulation headless")
@click.option("--imagery", is_flag=True, help="Show the cameras' imagery")
@click.option(
    "--population",
    type=click.Path(exists=True, dir_okay=False),
    help="Schema-v2 NPC population JSON.",
)
@click.option(
    "--semantics",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional semantic world JSON.",
)
def main(
    scene_xml_path: str | None,
    headless: bool,
    imagery: bool,
    population: str | None,
    semantics: str | None,
) -> None:
    """Launch the simulator and keep it alive until interrupted."""
    cameras_to_use = StretchCameras.all() if imagery else []
    sim = stretch_mujoco.StretchMujocoSimulator(
        scene_xml_path,
        cameras_to_use=cameras_to_use,
        population_path=population,
        semantic_world_path=semantics,
    )
    try:
        sim.start(headless=headless)
        while sim.is_running():
            if cameras_to_use:
                camera_data = sim.pull_camera_data()
                for camera in cameras_to_use:
                    cv2.imshow(camera.name, camera_data.get_camera_data(camera))
                cv2.waitKey(10)
    except KeyboardInterrupt:
        pass
    finally:
        if sim.is_running():
            sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
