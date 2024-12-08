<?php
$db = pg_connect("host=127.0.0.1 port=5432 dbname=healthte_drugbank user=healthte password=5#5)xMY1Y2Myyi");
if(isset($_POST['q'])){
// Fetch records
$search= $_POST['q'];
$query = "SELECT struct_id AS id, active_moiety_name AS text FROM public.active_ingredient WHERE active_moiety_name ILIKE '%$search%' LIMIT 6";
$query = pg_query($db, $query);
$data = array();
while ($row = pg_fetch_assoc($query)) {

   $id = $row['id'];
      $fullname = $row['text'];

      $data[] = array(
           "id" => $id, 
           "text" => $fullname
      );

}

echo json_encode($data);
die;
}
